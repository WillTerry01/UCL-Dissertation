# How to extract frames and prepare colmap

Using the **Black Mgic Camera App** you can simply lock the intrinics of the phone. The important settings are:
	- Shutter speed:1/100
	- Focal Length: locked, 0.7 seems to be a good average
	- ISO: 50
	- WB: Locked, set to what looks good in the current lighting
	- Camera Shake: OFF
	- FPS: 24 or 30
	- Zoom: 1x Zoom
	- Codex: H.264
	- Resolution: 4K (this can be changed to 720p for faster ORB SLAM Later)

With these setting you should get decent saved videos. Then to extract frames and intrinsics you need to install two packages, **exiftool** and **ffmpeg** using brew. 

Here are how to use these for extracting frames/ intrinsics:

exiftool frame0001.jpg



ffmpeg -i video.mp4 -qscale:v 2 -vf "fps=12" %06d.jpg

	-i video.mp4: your input video
	-qscale:v 2: sets image quality (lower = better, 1–3 is ideal)
	-vf "fps=10": extracts 10 frames per second from the video
	
frame_%04d.jpg: outputs images like frame_0001.jpg, frame_0002.jpg, etc.

# Edit

This did not actually work in the end, the camera extraction works very well but extracting the intrinsics does not. Use colmap for intrinsic extraction. Hopefully this will lead to better results.
