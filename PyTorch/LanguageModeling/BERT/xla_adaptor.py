import os
import torch

""" Support(XLA): xla_adaptor, only used in single card to debugging """
# XLA device
def is_xla_available():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        print("==== XLA is available ====")
        return xm.xla_device().type == 'xla'
    except Exception:
        print("XLA is unavailable!!! \n set GPU_NUM_DEVICES env ?")
        return False
IS_XLA_AVAILABLE = is_xla_available()

USE_XLA = (int(os.environ.get('USE_XLA')) and IS_XLA_AVAILABLE)
print("==== USE_XLA: {} ====".format(USE_XLA))

if USE_XLA:
    import torch_xla
    import torch_xla.core.xla_model as xm
    xla_device = xm.xla_device()
    xla_devices_list = torch_xla.core.xla_model.get_xla_supported_devices()
    print("==== XLA Device: ", xla_device)
    print("==== XLA Device List: ", xla_devices_list)
else:
    xla_device = None
    xla_devices_list = []
