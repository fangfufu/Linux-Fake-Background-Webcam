#!/bin/bash
gnome-terminal -- /bin/sh -c "node ./bodypix/app.js; exec bash"
sleep 1.5 
gnome-terminal -- /bin/sh -c "python3 ./fakecam/fake.py --no-foreground; exec bash" 
