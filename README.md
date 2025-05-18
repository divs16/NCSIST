Current Progress (as of May 18, 2025)
I extracted the largest mask from the SAM2-based segmentation output of a satellite image. The next step, as discussed, was to automate the process of generating a JSON prompt for object placement using the GPT API. However, I ran into issues setting up the correct libraries and importing the necessary modules to make API calls on my local machine.

As a temporary workaround, I manually uploaded the extracted 2D mask (corresponding to the largest contour) and labeled it. I plan to input this into GPT manually to get the JSON prompt for object placement.

Once I resolve the API configuration issues, I’ll automate this process and apply it to 4–5 more images as part of the full pipeline.

