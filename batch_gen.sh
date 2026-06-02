# python onnx_gen_DenseBlock.py --save-dir /data/nghiant/DenseBlock_mass --b 1,2,3,4,5,6,7,8 --c 4,8,12,16,20,24,28,32 --g 4,8,12,16,20,24,28,32 --input 32x32,64x64
# python dxnn_chain_compile.py --meta-path /data/nghiant/DenseBlock_mass/ --dxcom /home/nghiant/com/dx_com/dx_com

# python onnx_gen_MutationConv.py --save-dir /data/nghiant/MutationConv_mass --l 1,2,3,4,5,6,7,8 --c 4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64 --k 1,3,5,7 --input 32x32,64x64
# python dxnn_chain_compile.py --meta-path /data/nghiant/MutationConv_mass/ --dxcom /home/nghiant/com/dx_com/dx_com

# python onnx_gen_SqueezeExcitationBlock.py --save-dir /data/nghiant/SqueezeExcitationBlock_mass --b 1,2,3,4,5,6,7,8 --c 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128 --r 1,2,4,8 --input 32x32,64x64
# python dxnn_chain_compile.py --meta-path /data/nghiant/SqueezeExcitationBlock_mass/ --dxcom /home/nghiant/com/dx_com/dx_com

# python onnx_gen_StackingConv.py --save-dir /data/nghiant/StackingConv_mass --l 1,2,3,4,5,6,7,8 --ic 3,4 --f 8,16,24,32,40,48,56,64 --oc 3,4 --k 1,3 --input 32x32,64x64
# python dxnn_chain_compile.py --meta-path /data/nghiant/StackingConv_mass/ --dxcom /home/nghiant/com/dx_com/dx_com

# python onnx_gen_StackingLinear.py --save-dir /data/nghiant/StackingLinear_mass --l 1,2,3,4,5,6,7,8 --ic 8,16,32,64,128,256,512,1024 --f 1024,2048 --oc 8,16,32,64,128,256,512,1024
# python dxnn_chain_compile.py --meta-path /data/nghiant/StackingLinear_mass/ --dxcom /home/nghiant/com/dx_com/dx_com

python onnx_gen_TinyResidualBlock.py --save-dir /data/nghiant/TinyResidualBlock_mass --b 1,2,3,4,5,6,7,8 --c 32,64,96,128,160,192,224,256 --f 4,8,12,16,20,24,28,32 --input 32x32,64x64
python dxnn_chain_compile.py --meta-path /data/nghiant/TinyResidualBlock_mass/ --dxcom /home/nghiant/com/dx_com/dx_com

python onnx_gen_BottleneckResidualBlock.py --save-dir /data/nghiant/BottleneckResidualBlock_mass --b 1,2,3,4,5,6,7,8 --c 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128 --r 1,2,4,8 --input 32x32,64x64
python dxnn_chain_compile.py --meta-path /data/nghiant/BottleneckResidualBlock_mass/ --dxcom /home/nghiant/com/dx_com/dx_com

python onnx_gen_ResidualBlock.py --save-dir /data/nghiant/ResidualBlock_mass --b 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 --c 8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128 --input 32x32,64x64
python dxnn_chain_compile.py --meta-path /data/nghiant/ResidualBlock_mass/ --dxcom /home/nghiant/com/dx_com/dx_com
