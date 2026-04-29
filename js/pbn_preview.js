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

        // Create a container for the SVG
        const container = document.createElement("div");
        container.classList.add("pbn-svg-preview-container");
        container.style.width = "100%";
        container.style.height = "100%";
        container.style.background = "#ffffff";
        container.style.borderRadius = "4px";
        container.style.marginTop = "8px";
        container.style.display = "none"; // Hidden until first execution
        container.style.overflow = "hidden"; // Prevent scrollbars
        container.style.display = "flex";
        container.style.flexDirection = "column";

        this.pbn_svg_container = container;

        // Add as a DOM widget so it moves with the node
        const widget = this.addDOMWidget("pbn_svg_preview", "custom", container, {
          serialize: false,
          getValue() {
            return "";
          },
          setValue(v) {},
        });

        // Set up computeSize so the node knows how tall the widget is
        widget.computeSize = function (width) {
          return [width, this.computedHeight || 300];
        };

        this.pbn_svg_widget = widget;
      };

      // 2. Hook onExecuted to update the preview with new SVG data
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);

        if (message?.pbn_svg && Array.isArray(message.pbn_svg) && message.pbn_svg.length > 0) {
          // Show container and clear previous previews
          this.pbn_svg_container.style.display = "flex";
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
            img.style.height = "100%";
            img.style.objectFit = "contain";
            img.style.display = "block";
            img.alt = "PBN SVG Preview";

            // When image loads, adjust the node size to match aspect ratio
            img.onload = () => {
              const aspect = img.naturalHeight / img.naturalWidth;
              const width = this.size[0];
              const imgHeight = width * aspect;

              // Store calculated height on widget for computeSize
              this.pbn_svg_widget.computedHeight = imgHeight;

              // Recalculate node size based on widget content
              const size = this.computeSize([this.size[0], this.size[1]]);
              this.setSize([this.size[0], size[1]]);
              this.setDirtyCanvas(true, true);
            };

            this.pbn_svg_container.appendChild(img);
          });

          // Trigger initial canvas redraw
          this.setDirtyCanvas(true, true);
        }
      };
    }
  },
});
