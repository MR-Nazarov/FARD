# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""Network architectures, resolved by the conf's ``cnnModule`` field.

``MainModel.config_model`` does ``getattr(Nets, args.cnnModule)``, so an
architecture must be imported here to be reachable by a conf.

Included: FARD (this work) and the two baselines this work adapted -- MMMD-Net
extended to three contrasts, and EDSR+MMCHA adapted to post-reconstruction
fusion -- plus a MONAI 3D U-Net wrapper. Each names its origin in its header.

Baselines used unmodified are not vendored; see COMPARISON_METHODS.md.
"""

import nets.DnCNN
import nets.DnCNN_deb
import nets.Edsr
import nets.FARD_verFull
import nets.UNet3D
import nets.functions
import nets.main_model
