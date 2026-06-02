"""
mlperf inference benchmarking tool
"""
import argparse
import array
import json
import logging
import os
import sys
import threading
import time
from multiprocessing import JoinableQueue
import analyzer

SUPPORTED_PROFILES = {}
SUPPORTED_DATASETS = {}

# dataset list
from DatasetImageRandom import DatasetImageRandom

SUPPORTED_DATASETS["ImageRandom-test"] = (DatasetImageRandom, {})

# preprocessing
from preprocess import PreProcess, PreProcessUINT8RGBNCHW, PreProcessUINT8RGBNHWC

# postprocessing
from postprocess import PostProcess, PostProcessTorchAny, PostProcessDXM1Any, PostProcessHailo8Any

# backend list
try:
    from BackendDXM1 import BackendDXM1Sync, BackendDXM1Async

    SUPPORTED_PROFILES["dxm1a"] = (
        BackendDXM1Async, {},
        PreProcessUINT8RGBNHWC, {},
        PostProcessDXM1Any, {})

    SUPPORTED_PROFILES["dxm1s"] = (
        BackendDXM1Sync, {},
        PreProcessUINT8RGBNHWC, {},
        PostProcessDXM1Any, {})
except:
    print(f"[WARN] host machine does not support DX-M1")

try:
    from BackendHailo8 import BackendHailo8

    SUPPORTED_PROFILES["hailo8"] = (
        BackendHailo8, {},
        PreProcessUINT8RGBNHWC, {},
        PostProcessHailo8Any, {})
except:
    print(f"[WARN] host machine does not support Hailo8")

import mlperf_loadgen as lg
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Offline": lg.TestScenario.Offline,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIO_MAP.keys(), default="SingleStream", help="mlperf benchmark scenario")

    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--model-path", type=str, required=True, help="path to the model file")

    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", help="path to the dataset")
    parser.add_argument("--dataset-shape", type=str, help="input sample shape")
    parser.add_argument("--dataset-nums", type=int, help="number of samples")

    parser.add_argument("--output", help="test results")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--duration", default=0, type=int, help="duration in milliseconds (ms)")

    parser.add_argument("--performance-sample-count", type=int, help="performance sample count")
    parser.add_argument("--samples-per-query", default=1, type=int, help="query length (in terms of aggregated samples)")
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads for item-level parallel/concurrent/pipeline")
    parser.add_argument("--max-batchsize", type=int, default=8, help="max batch size in a single inference")

    args = parser.parse_args()

    return args

class Item:
    """An item that we queue for processing by the thread pool."""
    def __init__(self, query_id, content_id, xs):
        self.query_id = query_id
        self.content_id = content_id
        self.xs = xs

class RunnerBase:
    def __init__(self, backend, preprocessor, postprocessor, ds, threads, max_batchsize=128):
        self.ds = ds
        self.backend = backend
        self.pre_process = preprocessor
        self.post_process = postprocessor
        self.threads = threads
        self.max_batchsize = max_batchsize

    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, result_dict):
        self.result_dict = result_dict

        self.post_process.start()
        self.pre_process.start()
        self.backend.start()

    def run_one_item(self, qitem):
        # run the prediction
        processed_results = []
        try:
            # nghiant: preprocess inference postprocess
            # each of the object call to its own main processing function and do timing individually
            xs_preprocessed = self.pre_process(qitem.xs)
            results = self.backend(xs_preprocessed)
            processed_results = self.post_process(results, qitem.xs, xs_preprocessed)
            # nghiant_end

        except Exception as ex:  # pylint: disable=broad-except
            src = [self.ds.get_item_loc(i) for i in qitem.content_id]
            log.error("thread: failed on contentid=%s, %s", src, ex)
            # since post_process will not run, fake empty responses
            processed_results = [[]] * len(qitem.query_id)

        finally:
            lg.QuerySamplesComplete([lg.QuerySampleResponse(query_id, 0, 0) for query_id in qitem.query_id])

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            xs = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, xs))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                xs = self.ds.get_samples(idx[i:i+bs])
                self.run_one_item(Item(query_id[i:i+bs], idx[i:i+bs], xs))

    def finish(self):
        pass


class QueueRunner(RunnerBase):
    def __init__(self, backend, preprocessor, postprocessor, ds, threads, max_batchsize=128):
        super().__init__(backend, preprocessor, postprocessor, ds, threads, max_batchsize)
        self.tasks = JoinableQueue(maxsize=threads * 4)
        self.workers = []
        self.result_dict = {}

        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.tasks,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def handle_tasks(self, tasks_queue):
        """Worker thread."""
        while True:
            qitem = tasks_queue.get()
            if qitem is None:
                # None in the queue indicates the parent want us to exit
                tasks_queue.task_done()
                break
            self.run_one_item(qitem)
            tasks_queue.task_done()

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            xs = self.ds.get_samples(idx)
            self.tasks.put(Item(query_id, idx, xs))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                xs = self.ds.get_samples(idx[i:i+bs])
                self.tasks.put(Item(query_id[i:i+bs], idx[i:i+bs], xs))

    def finish(self):
        # exit all threads
        for _ in self.workers:
            self.tasks.put(None)
        for worker in self.workers:
            worker.join()


def add_results(final_results, name, result_dict, took):
    percentiles = [50., 80., 90., 95., 99., 99.9]

    preprocess_timing_list = [e-s for s, e in result_dict["preprocess_timing"]]
    preprocess_timing_q = np.percentile(preprocess_timing_list, percentiles).tolist()
    preprocess_timing_q_str = "  |  ".join([f"{p:4.1f}:{b*MILLI_SEC:8.3f}" for p, b in zip(percentiles, preprocess_timing_q)]) + f"  |  mean: {sum(preprocess_timing_list) / len(preprocess_timing_list) * MILLI_SEC:8.3f}"

    # analyzer.map_interval(result_dict["preprocess_timing"], name="preprocessing", visualize=True)

    inference_timing_list = [e-s for s, e in result_dict["inference_timing"]]
    inference_timing_q = np.percentile(inference_timing_list, percentiles).tolist()
    inference_timing_q_str = "  |  ".join([f"{p:4.1f}:{b*MILLI_SEC:8.3f}" for p, b in zip(percentiles, inference_timing_q)]) + f"  |  mean: {sum(inference_timing_list) / len(inference_timing_list) * MILLI_SEC:8.3f}"

    # analyzer.map_interval(result_dict["inference_timing"], name="inference", visualize=True)

    postprocess_timing_list = [e-s for s, e in result_dict["postprocess_timing"]]
    postprocess_timing_q = np.percentile(postprocess_timing_list, percentiles).tolist()
    postprocess_timing_q_str = "  |  ".join([f"{p:4.1f}:{b*MILLI_SEC:8.3f}" for p, b in zip(percentiles, postprocess_timing_q)]) + f"  |  mean: {sum(postprocess_timing_list) / len(postprocess_timing_list) * MILLI_SEC:8.3f}"

    # analyzer.map_interval(result_dict["postprocess_timing"], name="postprocessing", visualize=True)

    total_time_main_phases = sum(e-s for s, e in analyzer.map_interval(result_dict["preprocess_timing"] + result_dict["inference_timing"] + result_dict["postprocess_timing"]))

    analyzer.visualize_workload_density(
        [result_dict["preprocess_timing"], result_dict["inference_timing"], result_dict["postprocess_timing"], ],
        ["preprocessing", "inference", "postprocessing", ],
        ["r", "g", "b", ],
    )

    # this is what we record for each run
    result = {
        "total_time_main_phases": total_time_main_phases,
        "took": took,
        "preprocess_timing": {str(k): v for k, v in zip(percentiles, preprocess_timing_q)},
        "inference_timing": {str(k): v for k, v in zip(percentiles, inference_timing_q)},
        "postprocess_timing": {str(k): v for k, v in zip(percentiles, postprocess_timing_q)},
        "MLPerf_samples_per_second": result_dict["total"] / took,
        "samples_per_second": result_dict["total"] / total_time_main_phases,
        "total_items": len(inference_timing_list),
        "total_samples": result_dict["total"],
    }

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print(f'{name} samples={result_dict["total"]}, items={len(inference_timing_list)}\n\
        samples_per_second={result["MLPerf_samples_per_second"]:.2f}, time={took:.3f} (MLPerf)\n\
        samples_per_second={result["samples_per_second"]:.2f}, time={total_time_main_phases:.3f} (Only main phases)\n\
        -------------------------------------------------------------------------------------------------------------------------------------------------------------\n\
         preprocess_timing_tiles = {preprocess_timing_q_str}\n\
          inference_timing_tiles = {inference_timing_q_str}\n\
        postprocess_timing_tiles = {postprocess_timing_q_str}')



def main():
    args = get_args()

    log.info(args)

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    # backend
    backend_class, backend_settings, preprocess_class, preprocess_settings, postprocess_class, postprocess_settings = SUPPORTED_PROFILES[args.profile]
    backend = backend_class(**backend_settings)
    backend.load(args.model_path)
    
    preprocessor = preprocess_class(**preprocess_settings)
    postprocessor = postprocess_class(**postprocess_settings)

    # dataset
    dataset_class, dataset_settings = SUPPORTED_DATASETS[args.dataset]
    dataset_settings["shape"] = list(map(int, args.dataset_shape.split("x")))
    dataset_settings["n"] = args.dataset_nums
    dataset = dataset_class(**dataset_settings)

    # load model to backend
    final_results = {
        "runtime": backend.name(),
        "version": backend.version(),
        "time": time.perf_counter(),
        "cmdline": str(args),
    }

    count = dataset.get_item_count()
    # warmup
    dataset.load_query_samples([0])

    for _ in range(5):
        sample = dataset.get_samples([0])
        sample_preprocessed = preprocessor(sample)
        sample_inference = backend(sample_preprocessed)
        sample_postprocessed = postprocessor(sample_inference, sample, sample_preprocessed)

    dataset.unload_query_samples(None)

    # scenario
    scenario = SCENARIO_MAP[args.scenario]
    runner_map = {
        lg.TestScenario.SingleStream: RunnerBase,
        lg.TestScenario.MultiStream: QueueRunner,
        lg.TestScenario.Offline: QueueRunner,
    }

    runner = runner_map[scenario](backend, preprocessor, postprocessor, dataset, args.threads, max_batchsize=args.max_batchsize)

    def issue_queries(query_samples):
        runner.enqueue(query_samples)

    def flush_queries():
        pass

    settings = lg.TestSettings()
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly

    if args.duration:
        settings.min_duration_ms = args.duration
        settings.max_duration_ms = args.duration

    if scenario == lg.TestScenario.MultiStream:
        settings.multi_stream_samples_per_query = args.samples_per_query

    performance_sample_count = args.performance_sample_count if args.performance_sample_count else min(count, 100)

    if scenario == lg.TestScenario.SingleStream or scenario == lg.TestScenario.Offline:
        settings.min_query_count = performance_sample_count

    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(count, performance_sample_count, dataset.load_query_samples, dataset.unload_query_samples)

    log.info("starting {}".format(scenario))
    result_dict = {"total": 0, "psnr": 0.0, "ssim": 0.0, "scenario": str(scenario)}
    runner.start_run(result_dict)
    lg.StartTest(sut, qsl, settings)

    result_dict["total"] = runner.post_process.total

    preprocessor.finalize(result_dict)
    backend.finalize(result_dict)
    postprocessor.finalize(result_dict)

    add_results(final_results, "{}".format(scenario), result_dict, time.perf_counter() - dataset.last_loaded)

    runner.finish()
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    # write final results
    if args.output:
        with open("results.json", "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
