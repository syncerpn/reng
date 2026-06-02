import matplotlib.pyplot as plt

def map_interval(timing, name="", visualize=False):
    timing.sort()

    combined_intervals = []

    s, e = timing[0]
    for si, ei in timing:
        if si <= e:
            e = max(e, ei)
        else:
            combined_intervals.append((s, e))
            s, e = si, ei

    combined_intervals.append((s, e))

    if visualize:
        plt.subplots()
        plt.tight_layout()
        plt.xlabel("timestamp")
        plt.ylabel("interval index")
        plt.yticks([])

        tta = 0
        for i, (s, e) in enumerate(combined_intervals):
            tta += e - s
            plt.plot((s, e), (i, i), c="r" if i % 2 else "b")

        plt.title(f"measured by intervals: {tta/len(timing):.3f} s/item")
        plt.savefig(f"{name}.svg")
        plt.close()

    return combined_intervals

def visualize_workload_density(timing_list, name_list, color_list):
    plt.subplots(figsize=(10, 5))
    plt.tight_layout()
    plt.xlabel("timestamp")
    plt.ylabel("interval index")
    plt.yticks([])
    plt.ylim(0, len(name_list)+1)

    r = min([min([s for s, _ in timing]) for timing in timing_list])

    for i, (timing, color) in enumerate(zip(timing_list, color_list)):
        s = min([s for s, _ in timing])
        plt.scatter(s-r, i+1, s=4, c=color, alpha=1)

    plt.legend(name_list, loc="lower center", ncols=len(name_list))

    for i, (timing, color) in enumerate(zip(timing_list, color_list)):
        e = max([e for _, e in timing])
        plt.scatter(e-r, i+1, s=4, c=color, alpha=1)

    for i, (timing, name, color) in enumerate(zip(timing_list, name_list, color_list)):
        for s, e in timing:
            plt.plot((s-r, e-r), (i+1, i+1), linewidth=32, c=color, alpha=0.2, solid_capstyle='butt')

    plt.title(f"workload density")
    plt.savefig(f"workload.svg")
    plt.close()

