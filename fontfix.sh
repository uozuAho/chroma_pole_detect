#!/bin/bash
if [ ! -d /usr/share/fonts/truetype/dejavu ]; then
    sudo apt update
    sudo apt install fonts-dejavu-core
fi
mkdir -p .venv/lib/python3.14/site-packages/cv2/qt/fonts
cp /usr/share/fonts/truetype/dejavu/*.ttf \
   .venv/lib/python3.14/site-packages/cv2/qt/fonts/
