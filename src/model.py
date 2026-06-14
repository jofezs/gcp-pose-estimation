import torch.nn as nn
import torchvision.models as models


class GCPNet(nn.Module):
    """
    Shared CNN backbone (ResNet) with two heads:
      - keypoint head: regresses normalized (x, y) in [0, 1] via Sigmoid
      - classification head: 3-way shape classification (Cross/Square/L-Shape)

    Rationale: the marker location and its shape are correlated (the same
    visual region that tells you it's a "Cross" also tells you where its
    center is), so a shared backbone with light task-specific heads is more
    sample-efficient than two separate networks, while remaining simple to
    train and deploy.
    """

    def __init__(self, num_classes=3, backbone="resnet34", pretrained=True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        base = getattr(models, backbone)(weights=weights)

        # Drop the final avgpool + fc, keep conv feature extractor.
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = base.fc.in_features

        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.kp_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        f = self.shared(f)
        return self.kp_head(f), self.cls_head(f)
