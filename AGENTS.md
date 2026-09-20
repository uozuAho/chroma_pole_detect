# Light pole detection experiments

Python + opencv experiments to detect light poles in images and video. The light
pole is a vertical tower of 8 colored LEDs. The colors of the LEDs on the pole
can be green, pink or blue.

The ultimate goal of these experiments is to determine the algorithm most suited
for real-time localisation on an embedded computer running on a wheeled robot.
The light poles are used for the robot to locate itself. The target hardware is
described in ./docs/target-hardware.md . The eventual code will be written in C,
making the best use of the target hardware.

./detectors/ contains light pole detection algorithms.

./imgproc.py contains reusable image processing functions. Feel free to add to
this.

./img/ contains some example images of a light pole from various distances.

./img/real_centroids.json contains human-labeled centroids of each
image. These 'real centroids' aren't perfect, but are where a human would say
the center of the light pole is - roughly in the middle of the 8 LEDs.

./img/analysis.md contains some textual information about the images.

./scratchpads are for human use via a REPL


# Running the code
Run `just check` to run all linting and type checks.

Run `just measure` to measure all detectors on the example images. Results are
written to ./measurements
