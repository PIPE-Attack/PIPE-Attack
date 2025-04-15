#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")" # /home/ubuntu/PIPE-Attack/gan_attacks

# Train classification layer
cd  "$ROOT_DIR/BiDO" || exit
python train_reg.py --dataset=celeba --defense=reg --sameR d --prot fe

# Run GMI attack
cd "$ROOT_DIR/GMI" || exit
python attack.py --sameR d --prot fe

# Run KED-MI attack
cd "$ROOT_DIR/DMI" || exit
python k+1_gan_vib.py --dataset=celeba --defense=reg --sameR d --prot fe  # Train GAN
python recover_vib.py --dataset=celeba --defense=reg --sameR d --prot fe --verbose # Run KED-MI attack