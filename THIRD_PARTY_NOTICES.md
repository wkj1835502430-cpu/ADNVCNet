# Third-party notices

This repository uses external scientific-computing libraries and pretrained
visual models. Their code, model assets, and datasets retain their upstream
licenses and are not relicensed by this project.

| Project | Official upstream | License and use |
|---|---|---|
| DINOv2 | https://github.com/facebookresearch/dinov2 | The inspected repository license is Apache-2.0. The resource builder and inference code load an explicit external source checkout and weight file supplied by the user. DINOv2 implementation code and model weights are not bundled. Check the terms accompanying the particular model asset. |
| PyTorch | https://github.com/pytorch/pytorch | External dependency; retains its upstream license. |
| torchvision | https://github.com/pytorch/vision | External dependency used for ResNet-18 and its official ImageNet initialization; retains its upstream license and model-asset terms. |
| NumPy, OpenCV, Pillow, SciPy, scikit-learn, PyYAML | Their respective official projects and package distributions | Installed separately; no dependency binaries are bundled. Each distribution retains its own license. |

The paper also compares PatchCore and SimpleNet. The inspected upstream
versions were PatchCore commit
`fcaa92f124fb1ad74a7acf56726decd4b27cbcad` (Apache-2.0) and SimpleNet commit
`351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b` (MIT). Their source, trained weights,
and cached features are not included in this source-only package. Obtain them
from https://github.com/amazon-science/patchcore-inspection and
https://github.com/DonaldRR/SimpleNet when reproducing those baselines.

Unmodified copies of the previously inspected DINOv2, PatchCore, and SimpleNet
license texts are retained in `third_party_licenses/` for attribution and
reference. Inclusion of a license text does not imply inclusion of that
project's code or model assets.

MVTec AD and VisA must be obtained under their own dataset terms. The selected
examples visible in `paper/figures/` are attributed to these datasets and retain
the relevant dataset terms. The package supplies the manuscript figures,
computed response arrays, image-score rows, and small numerical plot inputs.
Complete dataset images, annotation masks, and model weights are obtained
separately. Normal-image identities and split indices apply the documented
dataset protocol; they are not image data.

These notices describe third-party materials. The license of the original
ADNVCNet project code is specified separately by its authors.
