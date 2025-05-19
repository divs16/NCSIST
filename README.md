##  Scene Generation Pipeline Overview

###  Goals
Automate the generation of simulation-ready `.json` files by:
- **Segmenting satellite images**
- **Deciding object placements using rules**
- **Outputting properly aligned Vega Prime scene configs**

---

##  Current Progress (as of May 18, 2025)

-  Extracted the **largest mask** from the **SAM2-based segmentation** output of a satellite image.
-  Intended next step: **Automate the generation of a JSON prompt** for object placement using the **GPT API**.
-  **Issue encountered**: Unable to configure the correct libraries and imports to make API calls locally.
-  **Temporary workaround**: 
  - Manually uploaded the **2D mask** corresponding to the **largest contour**.
  - Labeled it and plan to input it manually into GPT to receive a JSON prompt.
-  **Next steps**:
  - Resolve API configuration issues.
  - Automate the process end-to-end.
  - Apply the pipeline to **4–5 more images** to test consistency and scalability.
