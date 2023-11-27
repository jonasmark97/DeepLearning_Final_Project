import torch
import torchvision

if int(torchvision.__version__.split(".")[1]) >= 13:
    use_new_torchvision_load = True
else:
    use_new_torchvision_load = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassificationModel(torch.nn.Module):

    # Transfer learning model
    def __init__(self):
        super(ClassificationModel, self).__init__()

        # Load pre-trained model  
        if use_new_torchvision_load:
            self.model =  torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).to(device).eval()
        else:
            self.model = torchvision.models.vgg16(pretrained=True).to(device).eval()

        # Change the input size
        self.model.features[0] = torch.nn.Conv2d(
                                        in_channels=10,
                                        out_channels=64,
                                        kernel_size=(3,3),
                                        stride=(1,1),
                                        padding=(1,1)
                                    )

        # Change the ouput size
        self.model.classifier[6] = torch.nn.Linear(4096, 3, bias=True)


    def forward(self,x):
        return self.model.forward(x)