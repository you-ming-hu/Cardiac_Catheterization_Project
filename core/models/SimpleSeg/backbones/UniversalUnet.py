import segmentation_models_pytorch as smp

def create_module(encoder_name, encoder_weights, in_channels, out_channels):
        return smp.Unet(
            encoder_name = encoder_name, 
            encoder_weights = encoder_weights, 
            in_channels = in_channels, 
            classes = out_channels,
            activation = None,
            aux_params = None)