import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Paint By Number Node - SVG Vector Preview Extension
 *
 * This extension adds a custom DOM widget to the PaintByNumberNode to render
 * the generated SVG as a vector graphic using an <img> tag. This provides
 * a sharp, scalable preview alongside the pixel-based preview.
 */
app.registerExtension({
  name: "pbn.preview",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "PaintByNumberNode") {
      // 1. Hook onNodeCreated to initialize the preview container
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        // Create a scrollable container for the SVG(s)
        const container = document.createElement("div");
        container.classList.add("pbn-svg-preview-container");
        container.style.width = "100%";
        container.style.maxHeight = "500px";
        container.style.overflowY = "auto";
        container.style.background = "#ffffff";
        container.style.borderRadius = "4px";
        container.style.marginTop = "8px";
        container.style.display = "none"; // Hidden until first execution
        container.style.border = "1px solid #444";

        this.pbn_svg_container = container;

        // Add as a DOM widget so it moves with the node
        this.addDOMWidget("pbn_svg_preview", "custom", container, {
          serialize: false,
          getValue() {
            return "";
          },
          setValue(v) {},
        });
      };

      // 2. Hook onExecuted to update the preview with new SVG data
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);

        if (message?.pbn_svg && Array.isArray(message.pbn_svg) && message.pbn_svg.length > 0) {
          // Show container and clear previous previews
          this.pbn_svg_container.style.display = "block";
          this.pbn_svg_container.innerHTML = "";

          message.pbn_svg.forEach((svg) => {
            const img = document.createElement("img");

            // Construct the standard ComfyUI view URL
            const params = new URLSearchParams({
              filename: svg.filename,
              type: svg.type,
              subfolder: svg.subfolder,
              t: Date.now(), // Cache busting
            });

            // api.api_base is the base URL for the ComfyUI server
            img.src = api.api_base + "/view?" + params.toString();
            img.style.width = "100%";
            img.style.height = "auto";
            img.style.display = "block";
            img.style.padding = "4px";
            img.alt = "PBN SVG Preview";

            this.pbn_svg_container.appendChild(img);
          });

          // Trigger a canvas redraw to update the node height
          this.setDirtyCanvas(true, true);
        }
      };
    }
  },
});
