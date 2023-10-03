import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None

if found:
    from .scn import SpMiddleResNetFHD, SpEncoderDecoderFHD
    from .cylinder_backbone import Asymm_3d_spconv
else:
    print("No spconv, sparse convolution disabled!")

