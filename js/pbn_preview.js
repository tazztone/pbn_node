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
        // Subtle background for visibility
        container.style.background = "var(--comfy-input-bg, rgba(0, 0, 0, 0.1))";
        container.style.borderRadius = "4px";
        container.style.marginTop = "0px";
        container.style.display = "none"; // Hidden until first execution
        container.style.overflow = "hidden"; // Prevent scrollbars
        container.style.flexDirection = "column";

        // Create a single reusable img element
        const img = document.createElement("img");
        img.style.width = "100%";
        img.style.height = "100%";
        img.style.objectFit = "contain";
        img.style.display = "block";
        img.alt = "PBN SVG Preview";

        container.appendChild(img);
        this.pbn_svg_container = container;
        this.pbn_svg_img = img;

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

      // Handle horizontal node resizing smoothly
      const onResize = nodeType.prototype.onResize;
      nodeType.prototype.onResize = function (size) {
        onResize?.apply(this, arguments);
        if (this.pbn_svg_widget && this.pbn_svg_widget.aspectRatio) {
            const newHeight = Math.floor(size[0] * this.pbn_svg_widget.aspectRatio);
            this.pbn_svg_widget.computedHeight = newHeight;
        }
      };

      // 2. Hook onExecuted to update the preview with new SVG data
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);

        if (message?.pbn_svg && Array.isArray(message.pbn_svg) && message.pbn_svg.length > 0) {
          // Reset styling
          this.pbn_svg_container.style.display = "flex";
          this.pbn_svg_img.style.border = "none";
          this.pbn_svg_img.alt = "PBN SVG Preview";

          // Use only the first SVG to keep it simple
          const svg = message.pbn_svg[0];

          // Construct the standard ComfyUI view URL
          const params = new URLSearchParams({
            filename: svg.filename,
            type: svg.type,
            subfolder: svg.subfolder,
            t: Date.now(), // Cache busting
          });

          // When image loads, adjust the node size to match aspect ratio
          this.pbn_svg_img.onload = () => {
            const aspect = this.pbn_svg_img.naturalHeight / this.pbn_svg_img.naturalWidth;
            this.pbn_svg_widget.aspectRatio = aspect;
            const width = this.size[0];
            const imgHeight = Math.floor(width * aspect);

            // Only update if height changed significantly
            if (Math.abs((this.pbn_svg_widget.computedHeight || 0) - imgHeight) > 2) {
              this.pbn_svg_widget.computedHeight = imgHeight;

              // Recalculate node size
              const size = this.computeSize();
              this.setSize([this.size[0], size[1]]);
            }
            this.setDirtyCanvas(true, true);
          };

          this.pbn_svg_img.onerror = () => {
            this.pbn_svg_img.alt = "Failed to load SVG preview";
            this.pbn_svg_img.style.border = "1px solid red";
          };

          // Trigger image load
          this.pbn_svg_img.src = api.api_base + "/view?" + params.toString();
        }
      };
    }
  },
});
