import { app } from "../../../scripts/app.js";

const MASTER_NODE = "ComfyEverAnimateMasterSettings";
const INITIAL_CHUNK_NODE = "ComfyEverAnimateInitialChunk";
const EXTENSION_CHUNK_NODE = "ComfyEverAnimateContinueChunk";
const CHUNK_NODES = new Set([INITIAL_CHUNK_NODE, EXTENSION_CHUNK_NODE]);
const INITIAL_CUSTOM_WIDGETS = new Set(["startup_carry_frames"]);
const EXTENSION_CUSTOM_WIDGETS = new Set([
  "num_motion_latents",
  "continue_motion_max_frames",
  "motion_handoff_strength",
]);

const LEGACY_UI_ONLY_VALUES = new Set([
  "Model / Conditioning",
  "Video Settings",
  "Sampling Settings",
  "EverAnimate Settings",
  "Guide Inputs",
  "-",
  "not calculated",
]);

const MASTER_WIDGET_NAMES = [
  "ref_image_background",
  "width",
  "height",
  "chunk_length",
  "seed",
  "steps",
  "cfg",
  "sampler_name",
  "scheduler",
  "denoise",
  "num_video_anchor_latents",
  "pose_strength",
  "face_strength",
];

function isBooleanLike(value) {
  return typeof value === "boolean" || value === "true" || value === "false";
}

function sanitizeMasterWidgetValues(values) {
  if (!Array.isArray(values)) return values;
  const cleaned = values.filter((value) => value != null && !LEGACY_UI_ONLY_VALUES.has(value));
  const valuesOnly = cleaned.slice(0, MASTER_WIDGET_NAMES.length);

  if (valuesOnly.length >= 4 && !isBooleanLike(valuesOnly[0]) && isBooleanLike(valuesOnly[3])) {
    return [valuesOnly[3], valuesOnly[0], valuesOnly[1], valuesOnly[2], ...valuesOnly.slice(4)];
  }
  return valuesOnly;
}

function stripLegacyUiWidgets(node) {
  if (!node?.widgets) return;
  node.widgets = node.widgets.filter((widget) => {
    return !widget?.type?.startsWith?.("everanimate_")
      && !widget?.name?.startsWith?.("everanimate_header_")
      && widget?.name !== "extension_chunks_needed"
      && widget?.name !== "calculate chunks";
  });
}

function isAdvancedEnabled(node) {
  const widget = node?.widgets?.find((item) => item?.name === "enable_advanced");
  return widget?.value === true || widget?.value === "true";
}

function requestCanvasRedraw(node) {
  node?.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function customWidgetNamesForNode(node) {
  if (node?.comfyClass === INITIAL_CHUNK_NODE) return INITIAL_CUSTOM_WIDGETS;
  if (node?.comfyClass === EXTENSION_CHUNK_NODE) return EXTENSION_CUSTOM_WIDGETS;
  return new Set();
}

function setCustomWidgetsDisabled(node) {
  const customWidgetNames = customWidgetNamesForNode(node);
  if (!customWidgetNames.size) return;

  const disabled = !isAdvancedEnabled(node);
  for (const widget of node?.widgets ?? []) {
    if (!customWidgetNames.has(widget?.name)) continue;
    widget.disabled = disabled;
    if (widget.inputEl) {
      widget.inputEl.disabled = disabled;
      widget.inputEl.readOnly = disabled;
    }
  }
  requestCanvasRedraw(node);
}

function wrapAdvancedToggle(node) {
  const widget = node?.widgets?.find((item) => item?.name === "enable_advanced");
  if (!widget || widget._everAnimateCustomToggleWrapped) return;

  const originalCallback = widget.callback;
  widget.callback = function everAnimateCustomToggleCallback(...args) {
    const result = originalCallback?.apply(this, args);
    setCustomWidgetsDisabled(node);
    return result;
  };
  widget._everAnimateCustomToggleWrapped = true;
}

function installChunkCustomSettingsLock(nodeType, nodeData) {
  if (!CHUNK_NODES.has(nodeData.name) || nodeType.prototype._everAnimateCustomSettingsLockInstalled) return;

  const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function everAnimateChunkNodeCreated(...args) {
    const result = originalOnNodeCreated?.apply(this, args);
    wrapAdvancedToggle(this);
    setCustomWidgetsDisabled(this);
    return result;
  };

  const originalConfigure = nodeType.prototype.configure;
  nodeType.prototype.configure = function everAnimateChunkConfigure(...args) {
    const result = originalConfigure?.apply(this, args);
    wrapAdvancedToggle(this);
    setCustomWidgetsDisabled(this);
    return result;
  };

  nodeType.prototype._everAnimateCustomSettingsLockInstalled = true;
}

app.registerExtension({
  name: "ComfyEverAnimate.Cleanup",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    installChunkCustomSettingsLock(nodeType, nodeData);
    if (nodeData.name !== MASTER_NODE) return;

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function everAnimateConfigure(info) {
      if (info?.widgets_values) {
        info = { ...info, widgets_values: sanitizeMasterWidgetValues(info.widgets_values) };
      }
      const result = originalConfigure?.call(this, info);
      stripLegacyUiWidgets(this);
      return result;
    };

    const originalSerialize = nodeType.prototype.serialize;
    nodeType.prototype.serialize = function everAnimateSerialize(...args) {
      const data = originalSerialize?.apply(this, args);
      if (data?.widgets_values) {
        data.widgets_values = sanitizeMasterWidgetValues(data.widgets_values);
      }
      return data;
    };
  },
});
