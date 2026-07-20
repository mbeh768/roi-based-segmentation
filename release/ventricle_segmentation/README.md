If you can help it, try to do this whole process on a computer with a GPU and plenty of memory

---Systemwide Python Installation---
1. Download python 3.10 https://www.python.org/downloads/ - click download python install manager
2. answer Y for first three, answer N for online help

Python should now be added to PATH.

---Install Ventricle Segmentation Pipeline--
3. Run the "run_windows.bat" file

Opt in to GPU acceleration. CPU has less potential for complications but is much slower (I would *guess* a sample takes a few minutes on CPU and 30-60s on GPU)

It will create a python virtual environment, which will install the needed packages.

---Inference---
4. It will now prompt for a slide to image. 

I will include a tiff that you can test on labeled AJKY121_8-1.tiff which is a human heart that was in the validation set of training (mimics real performance).

When you click the image to segment, it will prompt for an ROI to be drawn in low resolution (we can work on this viewer).
There should be a popup with the various key binds ive set up to make segmentation easier.

MAKE SURE YOU CLICK 'f' TO FILL THE ROI REGION

The output (mask tiff file, image preview, and saved ROI file) will be saved in the image directory

Run 'run_windows.bat' again to do another image (you wont need to all the statup after the first time)

If any of this isn't easy just contact me because it will be much easier for me to fix things on my end than trying to troubleshoot.