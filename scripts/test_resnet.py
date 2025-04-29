import unittest
import torch
import torchvision
from lib.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class TestResNetVariants(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(1, 3, 224, 224)
        self.tolerance = 1e-5

    def compare_outputs(self, custom_model_class, tv_model_fn, num_classes):
        custom_model = custom_model_class(num_classes=num_classes, in_channels=3)
        custom_model.eval()

        tv_model = tv_model_fn(num_classes=num_classes)
        tv_model.eval()

        custom_model.load_from_torchvision(tv_model)
        custom_model.to(self.input_tensor.device)

        with torch.no_grad():
            custom_output = custom_model(self.input_tensor)
            tv_output = tv_model(self.input_tensor)

        max_diff = torch.abs(custom_output - tv_output).max().item()

        self.assertTrue(
            torch.allclose(custom_output, tv_output, atol=self.tolerance),
            msg=f"Max diff {max_diff:.6f} exceeds tolerance {self.tolerance}",
        )

    def test_resnet18(self):
        self.compare_outputs(ResNet18, torchvision.models.resnet18, num_classes=10)

    def test_resnet34(self):
        self.compare_outputs(ResNet34, torchvision.models.resnet34, num_classes=20)

    def test_resnet50(self):
        self.compare_outputs(ResNet50, torchvision.models.resnet50, num_classes=30)

    def test_resnet101(self):
        self.compare_outputs(ResNet101, torchvision.models.resnet101, num_classes=40)

    def test_resnet152(self):
        self.compare_outputs(ResNet152, torchvision.models.resnet152, num_classes=50)


if __name__ == "__main__":
    unittest.main()
