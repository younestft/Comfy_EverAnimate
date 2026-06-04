import { app } from "../../../scripts/app.js";

const MASTER_NODE = "ComfyEverAnimateMasterSettings";
const INITIAL_NODE = "ComfyEverAnimateInitialChunk";
const EXTENSION_NODE = "ComfyEverAnimateContinueChunk";
const DEFAULT_EXTENSION_CARRY_FRAMES = 5;

const SECTIONS = [
  ["model", "Model / Conditioning"],
  ["width", "Video Settings"],
  ["seed", "Sampling Settings"],
  ["num_video_anchor_latents", "EverAnimate Settings"],
  ["clip_vision_output", "Guide Inputs"],
];

function makeHeaderWidget(title) {
  return {
    name: `everanimate_header_${title}`,
    type: "everanimate_header",
    value: title,
    serialize: false,
    options: { serialize: false },
    draw(ctx, node, widgetWidth, y) {
      const margin = 10;
      ctx.save();
      ctx.strokeStyle = "#444";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin, y + 7);
      ctx.lineTo(widgetWidth - margin, y + 7);
      ctx.stroke();
      ctx.fillStyle = "#d8d8d8";
      ctx.font = "bold 12px Arial";
      ctx.textAlign = "left";
      ctx.fillText(title, margin, y + 25);
      ctx.restore();
    },
    computeSize(width) {
      return [width, 36];
    },
    serializeValue() {
      return undefined;
    },
  };
}

function installHeaders(node) {
  if (!node.widgets || node._everAnimateHeadersInstalled) return;
  node._everAnimateHeadersInstalled = true;

  node.widgets = node.widgets.filter((widget) => !widget.name?.startsWith("everanimate_header_"));

  for (const [beforeName, title] of [...SECTIONS].reverse()) {
    const index = node.widgets.findIndex((widget) => widget.name === beforeName);
    if (index >= 0) {
      node.widgets.splice(index, 0, makeHeaderWidget(title));
    }
  }

  requestAnimationFrame(() => {
    if (node.computeSize) {
      node.setSize(node.computeSize());
    }
    node.setDirtyCanvas(true, true);
  });
}

function widgetValue(node, name, fallback = 0) {
  const widget = node?.widgets?.find((item) => item.name === name);
  const value = Number(widget?.value);
  return Number.isFinite(value) ? value : fallback;
}

function inputLinkSourceNode(node, inputName) {
  const input = node.inputs?.find((item) => item.name === inputName);
  const linkId = input?.link;
  if (linkId == null) return null;
  const link = app.graph?.links?.[linkId];
  return link ? app.graph?.getNodeById?.(link.origin_id) : null;
}

function linkedInputNumericValue(node, inputName, fallback = 0, visited = new Set()) {
  const input = node?.inputs?.find((item) => item.name === inputName);
  const linkId = input?.link;
  if (linkId == null) return fallback;

  const link = app.graph?.links?.[linkId];
  const source = link ? app.graph?.getNodeById?.(link.origin_id) : null;
  if (!source || visited.has(source.id)) return fallback;
  visited.add(source.id);

  const directWidget = source.widgets?.find((item) => Number.isFinite(Number(item.value)));
  if (directWidget) return Number(directWidget.value);

  for (const value of source.widgets_values ?? []) {
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }

  if (source.inputs?.length) {
    for (const sourceInput of source.inputs) {
      const value = linkedInputNumericValue(source, sourceInput.name, Number.NaN, visited);
      if (Number.isFinite(value)) return value;
    }
  }

  return fallback;
}

function firstOutputTargetNode(node, outputName, comfyClass) {
  const output = node.outputs?.find((item) => item.name === outputName);
  for (const linkId of output?.links ?? []) {
    const link = app.graph?.links?.[linkId];
    const target = link ? app.graph?.getNodeById?.(link.target_id) : null;
    if (!comfyClass || target?.comfyClass === comfyClass) {
      return target;
    }
  }
  return null;
}

function hasLinkedInput(node, inputName) {
  const input = node?.inputs?.find((item) => item.name === inputName);
  return input?.link != null;
}

function trimAmountFromFrames(frameCount) {
  if (frameCount <= 0) return 0;
  const latentCount = Math.floor((Math.max(1, frameCount) - 1) / 4) + 1;
  return Math.max(0, latentCount * 4 - 3);
}

function calculateExtensionChunksNeeded(node) {
  const master = inputLinkSourceNode(node, "initial_settings");
  if (!master || master.comfyClass !== MASTER_NODE) {
    return { value: "-", hint: "connect master" };
  }

  const frameCount = Math.floor(linkedInputNumericValue(master, "frame_count", 0));
  if (frameCount <= 0) {
    return { value: "-", hint: "connect frame count" };
  }

  const chunkLength = Math.max(1, Math.floor(widgetValue(master, "chunk_length", 81)));
  const startupCarryFrames = Math.max(0, Math.floor(widgetValue(node, "startup_carry_frames", 1)));
  const extension = firstOutputTargetNode(node, "settings", EXTENSION_NODE);
  const extensionCarryFrames = Math.max(
    1,
    Math.floor(widgetValue(extension, "continue_motion_max_frames", DEFAULT_EXTENSION_CARRY_FRAMES)),
  );

  const hasReferenceImage = hasLinkedInput(master, "reference_image");
  const startupTrim = hasReferenceImage && startupCarryFrames > 0 ? trimAmountFromFrames(startupCarryFrames) : 0;
  const extensionTrim = trimAmountFromFrames(extensionCarryFrames);
  const firstKept = Math.max(1, chunkLength - startupTrim);
  const extensionKept = Math.max(1, chunkLength - extensionTrim);
  const needed = frameCount <= firstKept ? 0 : Math.ceil((frameCount - firstKept) / extensionKept);

  return {
    value: String(needed),
    hint: `${frameCount} frames`,
  };
}

function makeExtensionChunksNeededResultWidget() {
  return {
    name: "extension_chunks_needed",
    type: "everanimate_result",
    value: "-",
    hint: "not calculated",
    serialize: false,
    options: { serialize: false },
    draw(ctx, node, widgetWidth, y) {
      const margin = 10;
      const value = this.value ?? "-";
      const hint = this.hint ?? "not calculated";
      ctx.save();
      ctx.fillStyle = "#9a9a9a";
      ctx.font = "12px Arial";
      ctx.textAlign = "left";
      ctx.fillText("extension chunks needed", margin, y + 17);
      ctx.fillStyle = "#d8d8d8";
      ctx.font = "11px Arial";
      ctx.fillText(hint, margin, y + 34);
      ctx.fillStyle = "#e8e8e8";
      ctx.font = "bold 20px Arial";
      ctx.textAlign = "right";
      ctx.fillText(String(value), widgetWidth - margin, y + 25);
      ctx.restore();
    },
    computeSize(width) {
      return [width, 42];
    },
    serializeValue() {
      return undefined;
    },
  };
}

function installExtensionChunksNeeded(node) {
  if (!node.widgets || node._everAnimateExtensionChunksInstalled) return;
  node._everAnimateExtensionChunksInstalled = true;

  node.widgets = node.widgets.filter((widget) => widget.name !== "extension_chunks_needed");
  const button = node.addWidget("button", "calculate chunks", null, () => {
    const result = calculateExtensionChunksNeeded(node);
    const widget = node.widgets?.find((item) => item.name === "extension_chunks_needed");
    if (widget) {
      widget.value = result.value;
      widget.hint = result.hint;
    }
    node.setDirtyCanvas(true, true);
  });
  button.serialize = false;
  button.options = { ...(button.options ?? {}), serialize: false };
  node.widgets.push(makeExtensionChunksNeededResultWidget());

  requestAnimationFrame(() => {
    if (node.computeSize) {
      node.setSize(node.computeSize());
    }
    node.setDirtyCanvas(true, true);
  });
}

app.registerExtension({
  name: "ComfyEverAnimate.UI",
  async nodeCreated(node) {
    if (node.comfyClass === MASTER_NODE) {
      setTimeout(() => installHeaders(node), 0);
    }
    if (node.comfyClass === INITIAL_NODE) {
      setTimeout(() => installExtensionChunksNeeded(node), 0);
    }
  },
});
