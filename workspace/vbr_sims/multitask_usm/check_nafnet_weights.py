import sys, torch
sys.path.insert(0, r'E:\dev\Code\Python\bones\enhance\NAFNet')
weight_path = r'E:\dev\Code\Python\bones\enhance\NAFNet\experiments\pretrained_models\NAFNet-GoPro-width32.pth'
ckpt = torch.load(weight_path, map_location='cpu')
print(f'Checkpoint type: {type(ckpt)}')
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print(f'Top-level keys: {keys[:10]}')
    if 'params' in ckpt:
        params = ckpt['params']
        print(f'params type: {type(params)}, len: {len(params)}')
        if isinstance(params, dict):
            pkeys = list(params.keys())
            print(f'First 5 param keys: {pkeys[:5]}')
            print(f'Last 5 param keys: {pkeys[-5:]}')
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        sdkeys = list(sd.keys())
        print(f'state_dict len: {len(sdkeys)}')
        print(f'First 5: {sdkeys[:5]}')
    else:
        print(f'Direct dict, first 5 keys: {keys[:5]}')
        # Try to load directly
        from basicsr.models.archs.NAFNet_arch import NAFNet
        model = NAFNet(img_channel=3, width=32, middle_blk_num=1,
                       enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])
        try:
            model.load_state_dict(ckpt, strict=True)
            print('Direct state_dict load: SUCCESS')
        except Exception as e:
            print(f'Direct load failed: {e}')
