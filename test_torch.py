try:
    import torch
    print('✅ PyTorch version:', torch.__version__)
    print('✅ PyTorch working!')
    print('CUDA available:', torch.cuda.is_available())
except Exception as e:
    print('❌ PyTorch error:', e)
    print('❌ PyTorch is not working properly')