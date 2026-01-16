import timm
from src.models import vgg

def test_load_timm_weights_vgg16():
    timm_model = timm.create_model('vgg16', pretrained=True, num_classes=10)
    model = vgg.VGG(
        num_classes=10,
        cfg=vgg._cfgs['vgg16'],
    )

    state_dict = timm_model.state_dict()
    model.load_state_dict(state_dict, strict=True)
