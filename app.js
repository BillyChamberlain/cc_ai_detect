"use strict";

const CONFIG = {
  FACE_MODEL_PATH: "./models/face_best.onnx",
  PHONE_MODEL_PATH: "./models/best.onnx",

  MODEL_INPUT_SIZE: 640,

  FACE_CLASS_ID: 0,
  PHONE_CLASS_ID: 0,

  FACE_CONFIDENCE: 0.45,
  PHONE_CONFIDENCE: 0.45,
  NMS_IOU_THRESHOLD: 0.45,

  INFERENCE_INTERVAL_MS: 125,

  FOG_DELAY_AFTER_PHONE_MS: 20_000,

  MIRROR_CAMERA: true,

  FACE_HOLD_MS: 500,
  PHONE_HOLD_MS: 420,

  FACE_LOCK_DURATION_MS: 700,

  ORBIT_BOX_WIDTH: 172,
  ORBIT_BOX_HEIGHT: 120,
  ORBIT_RADIUS_RATIO: 0.18,
  ORBIT_SPEED: 0.00105,
  ORBIT_Y_RATIO: 1,

  FRONT_BOX_SCALE: 1.08,

  BOX_SMOOTHING: 0.1,

  PRIVACY_BLUR_AMOUNT: 36,
  PRIVACY_BLUR_PASSES: 2,
  PRIVACY_GRAYSCALE: 1,
  PRIVACY_CONTRAST: 1.8,
  PRIVACY_BRIGHTNESS: 0.82,
  EDGE_SOFTNESS: 22,

  FOG_SPAWN_INTERVAL_MS: 155,
  MAX_FOG_PUFFS: 250,
};

const video = document.getElementById("camera");
const visualCanvas = document.getElementById("visualCanvas");
const inferenceCanvas = document.getElementById("inferenceCanvas");

const visualCtx = visualCanvas.getContext("2d");
const inferenceCtx = inferenceCanvas.getContext("2d", {
  willReadFrequently: true,
});

const blurCanvas = document.createElement("canvas");
const blurCtx = blurCanvas.getContext("2d");

const intro = document.getElementById("intro");
const startButton = document.getElementById("startButton");
const phaseLabel = document.getElementById("phaseLabel");
const modelLabel = document.getElementById("modelLabel");

const state = {
  started: false,
  startTime: 0,

  cameraStarted: false,
  cameraStarting: false,

  faceModelLoading: false,
  phoneModelLoading: false,

  faceSession: null,
  phoneSession: null,

  faceInputName: null,
  phoneInputName: null,

  lastInferenceTime: 0,

  face: null,
  phone: null,

  lastFaceSeenAt: 0,
  lastPhoneSeenAt: 0,

  faceLockStartedAt: null,
  wasFaceActive: false,

  phoneTriggerTime: null,

  /*
    Stage 1: Face/orbit behavior.
    Stage 2: Phone response.
    Stage 3: Permanent fog.
  */
  stage: 1,

  boxX: window.innerWidth / 2,
  boxY: window.innerHeight / 2,
  boxWidth: CONFIG.ORBIT_BOX_WIDTH,
  boxHeight: CONFIG.ORBIT_BOX_HEIGHT,

  fogPuffs: [],
  lastFogSpawnAt: 0,

  width: window.innerWidth,
  height: window.innerHeight,
};

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  state.width = window.innerWidth;
  state.height = window.innerHeight;

  visualCanvas.width = Math.floor(state.width * dpr);
  visualCanvas.height = Math.floor(state.height * dpr);

  visualCanvas.style.width = `${state.width}px`;
  visualCanvas.style.height = `${state.height}px`;

  visualCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function getElapsedMs() {
  if (!state.startTime) return 0;
  return performance.now() - state.startTime;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(current, target, amount) {
  return current + (target - current) * amount;
}

function randomRange(min, max) {
  return min + Math.random() * (max - min);
}

function isFaceActive(now) {
  return (
    state.face !== null &&
    now - state.lastFaceSeenAt <= CONFIG.FACE_HOLD_MS
  );
}

function isPhoneActive(now) {
  return (
    state.phone !== null &&
    now - state.lastPhoneSeenAt <= CONFIG.PHONE_HOLD_MS
  );
}

function updatePersistentStage(now) {
  if (!state.started) {
    state.stage = 1;
    return;
  }

  if (
    state.stage < 2 &&
    state.phoneTriggerTime !== null
  ) {
    state.stage = 2;
  }

  if (
    state.stage === 2 &&
    state.phoneTriggerTime !== null &&
    now - state.phoneTriggerTime >=
      CONFIG.FOG_DELAY_AFTER_PHONE_MS
  ) {
    state.stage = 3;
  }
}

function getCurrentStage() {
  return state.stage;
}

function getVideoCoverRect() {
  const videoWidth = video.videoWidth || 1;
  const videoHeight = video.videoHeight || 1;

  const screenAspect = state.width / state.height;
  const videoAspect = videoWidth / videoHeight;

  let width;
  let height;
  let x;
  let y;

  if (videoAspect > screenAspect) {
    height = state.height;
    width = height * videoAspect;
    x = (state.width - width) / 2;
    y = 0;
  } else {
    width = state.width;
    height = width / videoAspect;
    x = 0;
    y = (state.height - height) / 2;
  }

  return { x, y, width, height };
}

function mapModelPointToScreen(modelX, modelY) {
  const videoRect = getVideoCoverRect();

  return {
    x:
      videoRect.x +
      (modelX / CONFIG.MODEL_INPUT_SIZE) *
        videoRect.width,

    y:
      videoRect.y +
      (modelY / CONFIG.MODEL_INPUT_SIZE) *
        videoRect.height,
  };
}

function getFaceScreenData() {
  if (!state.face) return null;

  const center = mapModelPointToScreen(
    state.face.x,
    state.face.y
  );

  const videoRect = getVideoCoverRect();

  const screenFaceWidth =
    (state.face.width / CONFIG.MODEL_INPUT_SIZE) *
    videoRect.width;

  const screenFaceHeight =
    (state.face.height / CONFIG.MODEL_INPUT_SIZE) *
    videoRect.height;

  return {
    x: center.x,
    y: center.y,
    width: Math.max(56, screenFaceWidth),
    height: Math.max(56, screenFaceHeight),
  };
}

async function createOrtSession(modelPath) {
  try {
    return await ort.InferenceSession.create(modelPath, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
  } catch (webGpuError) {
    console.warn(
      "WebGPU unavailable. Falling back to ONNX Runtime WASM.",
      webGpuError
    );

    return ort.InferenceSession.create(modelPath, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
}

async function loadFaceModel() {
  if (state.faceSession || state.faceModelLoading) {
    return;
  }

  state.faceModelLoading = true;

  try {
    state.faceSession = await createOrtSession(
      CONFIG.FACE_MODEL_PATH
    );

    state.faceInputName =
      state.faceSession.inputNames[0];
  } finally {
    state.faceModelLoading = false;
  }
}

async function loadPhoneModel() {
  if (state.phoneSession || state.phoneModelLoading) {
    return;
  }

  state.phoneModelLoading = true;

  try {
    state.phoneSession = await createOrtSession(
      CONFIG.PHONE_MODEL_PATH
    );

    state.phoneInputName =
      state.phoneSession.inputNames[0];
  } finally {
    state.phoneModelLoading = false;
  }
}

async function startCamera() {
  if (state.cameraStarted || state.cameraStarting) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      "Camera access requires localhost or HTTPS."
    );
  }

  state.cameraStarting = true;

  try {
    const stream =
      await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = resolve;
    });

    await video.play();

    state.cameraStarted = true;
    video.classList.add("visible");
  } finally {
    state.cameraStarting = false;
  }
}

async function beginExperience() {
  if (state.started) return;

  state.started = true;
  state.startTime = performance.now();
  state.stage = 1;

  /*
    Remove the intro screen automatically if it still exists.
  */
  if (intro) {
    intro.classList.add("hidden");
  }

  if (startButton) {
    startButton.disabled = true;
  }

  phaseLabel.textContent =
    "Stage 01 · Try bringing a phone into the frame";

  modelLabel.textContent = "Opening camera";

  try {
    await Promise.all([
      startCamera(),
      loadFaceModel(),
      loadPhoneModel(),
    ]);

    modelLabel.textContent = "Searching for face";
  } catch (error) {
    console.error("Camera or model startup failed:", error);
    modelLabel.textContent = "Camera or model error";
  }
}

function drawVideoToInferenceCanvas() {
  const size = CONFIG.MODEL_INPUT_SIZE;

  inferenceCanvas.width = size;
  inferenceCanvas.height = size;

  inferenceCtx.clearRect(0, 0, size, size);
  inferenceCtx.fillStyle = "#000";
  inferenceCtx.fillRect(0, 0, size, size);

  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  if (!videoWidth || !videoHeight) return;

  const scale = Math.min(
    size / videoWidth,
    size / videoHeight
  );

  const drawWidth = videoWidth * scale;
  const drawHeight = videoHeight * scale;
  const offsetX = (size - drawWidth) / 2;
  const offsetY = (size - drawHeight) / 2;

  if (CONFIG.MIRROR_CAMERA) {
    inferenceCtx.save();
    inferenceCtx.translate(size, 0);
    inferenceCtx.scale(-1, 1);

    inferenceCtx.drawImage(
      video,
      offsetX,
      offsetY,
      drawWidth,
      drawHeight
    );

    inferenceCtx.restore();
  } else {
    inferenceCtx.drawImage(
      video,
      offsetX,
      offsetY,
      drawWidth,
      drawHeight
    );
  }
}

function createInputTensor() {
  const size = CONFIG.MODEL_INPUT_SIZE;
  const imageData = inferenceCtx.getImageData(
    0,
    0,
    size,
    size
  );

  const pixels = imageData.data;
  const channelSize = size * size;
  const input = new Float32Array(
    3 * channelSize
  );

  for (
    let pixelIndex = 0;
    pixelIndex < channelSize;
    pixelIndex++
  ) {
    const sourceIndex = pixelIndex * 4;

    input[pixelIndex] =
      pixels[sourceIndex] / 255;

    input[channelSize + pixelIndex] =
      pixels[sourceIndex + 1] / 255;

    input[channelSize * 2 + pixelIndex] =
      pixels[sourceIndex + 2] / 255;
  }

  return new ort.Tensor(
    "float32",
    input,
    [1, 3, size, size]
  );
}

function calculateIoU(first, second) {
  const firstLeft =
    first.x - first.width / 2;

  const firstTop =
    first.y - first.height / 2;

  const firstRight =
    first.x + first.width / 2;

  const firstBottom =
    first.y + first.height / 2;

  const secondLeft =
    second.x - second.width / 2;

  const secondTop =
    second.y - second.height / 2;

  const secondRight =
    second.x + second.width / 2;

  const secondBottom =
    second.y + second.height / 2;

  const overlapLeft = Math.max(
    firstLeft,
    secondLeft
  );

  const overlapTop = Math.max(
    firstTop,
    secondTop
  );

  const overlapRight = Math.min(
    firstRight,
    secondRight
  );

  const overlapBottom = Math.min(
    firstBottom,
    secondBottom
  );

  const overlapWidth = Math.max(
    0,
    overlapRight - overlapLeft
  );

  const overlapHeight = Math.max(
    0,
    overlapBottom - overlapTop
  );

  const overlapArea =
    overlapWidth * overlapHeight;

  const firstArea =
    (firstRight - firstLeft) *
    (firstBottom - firstTop);

  const secondArea =
    (secondRight - secondLeft) *
    (secondBottom - secondTop);

  return (
    overlapArea /
    (
      firstArea +
      secondArea -
      overlapArea +
      1e-6
    )
  );
}

function nonMaximumSuppression(detections) {
  const candidates = [...detections].sort(
    (a, b) => b.confidence - a.confidence
  );

  const selected = [];

  while (candidates.length > 0) {
    const best = candidates.shift();
    selected.push(best);

    for (
      let index = candidates.length - 1;
      index >= 0;
      index--
    ) {
      if (
        calculateIoU(
          best,
          candidates[index]
        ) > CONFIG.NMS_IOU_THRESHOLD
      ) {
        candidates.splice(index, 1);
      }
    }
  }

  return selected;
}

function parseYoloOutput(
  outputTensor,
  wantedClassId,
  confidenceThreshold
) {
  const { data, dims } = outputTensor;

  /*
    ONNX export with nms=True:
    [1, detectionCount, 6]

    Detection:
    [x1, y1, x2, y2, confidence, classId]
  */
  if (
    dims.length === 3 &&
    dims[2] === 6
  ) {
    const detections = [];
    const count = dims[1];

    for (
      let index = 0;
      index < count;
      index++
    ) {
      const offset = index * 6;

      const x1 = data[offset];
      const y1 = data[offset + 1];
      const x2 = data[offset + 2];
      const y2 = data[offset + 3];
      const confidence = data[offset + 4];
      const classId = Math.round(
        data[offset + 5]
      );

      if (
        classId === wantedClassId &&
        confidence >= confidenceThreshold
      ) {
        detections.push({
          x: (x1 + x2) / 2,
          y: (y1 + y2) / 2,
          width: x2 - x1,
          height: y2 - y1,
          confidence,
        });
      }
    }

    return (
      nonMaximumSuppression(detections)[0] ||
      null
    );
  }

  /*
    Raw YOLO output:
    [1, 4 + classCount, anchorCount]
    or
    [1, anchorCount, 4 + classCount]
  */
  if (dims.length !== 3) {
    console.warn(
      "Unexpected YOLO output dimensions:",
      dims
    );

    return null;
  }

  const attributesFirst =
    dims[1] < dims[2];

  const attributes = attributesFirst
    ? dims[1]
    : dims[2];

  const anchors = attributesFirst
    ? dims[2]
    : dims[1];

  if (attributes < 5) {
    console.warn(
      "Invalid YOLO output shape:",
      dims
    );

    return null;
  }

  const detections = [];

  for (
    let anchor = 0;
    anchor < anchors;
    anchor++
  ) {
    const read = (attribute) => {
      if (attributesFirst) {
        return data[
          attribute * anchors + anchor
        ];
      }

      return data[
        anchor * attributes + attribute
      ];
    };

    const x = read(0);
    const y = read(1);
    const width = read(2);
    const height = read(3);

    let highestScore = -Infinity;
    let detectedClassId = -1;

    for (
      let attribute = 4;
      attribute < attributes;
      attribute++
    ) {
      const score = read(attribute);

      if (score > highestScore) {
        highestScore = score;
        detectedClassId = attribute - 4;
      }
    }

    if (
      detectedClassId === wantedClassId &&
      highestScore >= confidenceThreshold
    ) {
      detections.push({
        x,
        y,
        width,
        height,
        confidence: highestScore,
      });
    }
  }

  return (
    nonMaximumSuppression(detections)[0] ||
    null
  );
}

async function runModel(
  session,
  inputName,
  wantedClassId,
  confidenceThreshold,
  inputTensor
) {
  const outputs = await session.run({
    [inputName]: inputTensor,
  });

  const outputTensor =
    outputs[session.outputNames[0]];

  return parseYoloOutput(
    outputTensor,
    wantedClassId,
    confidenceThreshold
  );
}

async function detect(now) {
  if (!state.started || !state.cameraStarted) {
    return;
  }

  if (!state.faceSession || !video.videoWidth) {
    return;
  }

  if (
    now - state.lastInferenceTime <
    CONFIG.INFERENCE_INTERVAL_MS
  ) {
    return;
  }

  state.lastInferenceTime = now;

  try {
    drawVideoToInferenceCanvas();

    const inputTensor = createInputTensor();

    const faceDetection = await runModel(
      state.faceSession,
      state.faceInputName,
      CONFIG.FACE_CLASS_ID,
      CONFIG.FACE_CONFIDENCE,
      inputTensor
    );

    if (faceDetection) {
      state.face = faceDetection;
      state.lastFaceSeenAt = now;
    }

    if (state.phoneSession) {
      const phoneDetection = await runModel(
        state.phoneSession,
        state.phoneInputName,
        CONFIG.PHONE_CLASS_ID,
        CONFIG.PHONE_CONFIDENCE,
        inputTensor
      );

      if (phoneDetection) {
        state.phone = phoneDetection;
        state.lastPhoneSeenAt = now;

        if (state.phoneTriggerTime === null) {
          state.phoneTriggerTime = now;
          state.stage = 2;
        }
      }
    }
  } catch (error) {
    console.error("Inference error:", error);
    modelLabel.textContent = "Inference paused";
  }
}

function updateStageUi(now) {
  const stage = getCurrentStage();

  if (stage === 1) {
    phaseLabel.textContent =
      "Stage 01 · Try bringing a phone into the frame";

    if (state.cameraStarting) {
      modelLabel.textContent = "Opening camera";
    } else if (!state.cameraStarted) {
      modelLabel.textContent = "Camera waiting";
    } else if (isFaceActive(now)) {
      modelLabel.textContent = "Face detected";
    } else {
      modelLabel.textContent = "Searching for face";
    }

    return;
  }

  if (stage === 2) {
    phaseLabel.textContent = "Stage 02";
    modelLabel.textContent = isPhoneActive(now)
      ? "Phone detected"
      : "Phone response entered";

    return;
  }

  if (stage === 3) {
    phaseLabel.textContent = "Stage 03";
    modelLabel.textContent = isPhoneActive(now)
      ? "?"
      : "?";
  }
}

function updateBlurBox(now) {
  const elapsed = getElapsedMs();
  const stage = getCurrentStage();

  let targetX = state.width / 2;
  let targetY = state.height / 2;
  let targetWidth = CONFIG.ORBIT_BOX_WIDTH;
  let targetHeight = CONFIG.ORBIT_BOX_HEIGHT;

  const faceActive = isFaceActive(now);

  if (
    faceActive &&
    !state.wasFaceActive
  ) {
    state.faceLockStartedAt = now;
  }

  if (!faceActive) {
    state.faceLockStartedAt = null;
  }

  state.wasFaceActive = faceActive;

  /*
    Stage 1:
    Face-following orbit behavior.
  */
  if (stage === 1) {
    if (faceActive) {
      const face = getFaceScreenData();

      const faceLockAge =
        state.faceLockStartedAt === null
          ? Infinity
          : now - state.faceLockStartedAt;

      const isInFaceLock =
        faceLockAge <
        CONFIG.FACE_LOCK_DURATION_MS;

      if (isInFaceLock) {
        targetX = face.x;
        targetY = face.y;
        targetWidth = face.width * 1.15;
        targetHeight = face.height * 1.18;
      } else {
        const orbitRadius = Math.max(
          Math.max(
            face.width,
            face.height
          ) * 1.45,

          Math.min(
            state.width,
            state.height
          ) * CONFIG.ORBIT_RADIUS_RATIO
        );

        const orbitAngle =
          elapsed * CONFIG.ORBIT_SPEED;

        targetX =
          face.x +
          Math.cos(orbitAngle) *
          orbitRadius;

        targetY =
          face.y +
          Math.sin(orbitAngle) *
          orbitRadius *
          CONFIG.ORBIT_Y_RATIO;

        targetWidth = CONFIG.ORBIT_BOX_WIDTH;
        targetHeight = CONFIG.ORBIT_BOX_HEIGHT;
      }
    } else {
      targetX =
        state.width * 0.5 +
        Math.cos(elapsed * 0.00047) *
        state.width *
        0.12;

      targetY =
        state.height * 0.46 +
        Math.sin(elapsed * 0.00062) *
        44;

      targetWidth = CONFIG.ORBIT_BOX_WIDTH;
      targetHeight = CONFIG.ORBIT_BOX_HEIGHT;
    }
  }

  /*
    Stage 2 and Stage 3:
    Cover the face while the phone is visible.
    Otherwise return to the orbit.
  */
  if (stage === 2 || stage === 3) {
    if (faceActive) {
      const face = getFaceScreenData();

      if (isPhoneActive(now)) {
        targetX = face.x;
        targetY = face.y;

        targetWidth =
          face.width *
          CONFIG.FRONT_BOX_SCALE;

        targetHeight =
          face.height *
          CONFIG.FRONT_BOX_SCALE;
      } else {
        const orbitRadius = Math.max(
          Math.max(
            face.width,
            face.height
          ) * 1.45,

          Math.min(
            state.width,
            state.height
          ) * CONFIG.ORBIT_RADIUS_RATIO
        );

        const orbitAngle =
          elapsed * CONFIG.ORBIT_SPEED;

        targetX =
          face.x +
          Math.cos(orbitAngle) *
          orbitRadius;

        targetY =
          face.y +
          Math.sin(orbitAngle) *
          orbitRadius *
          CONFIG.ORBIT_Y_RATIO;

        targetWidth = CONFIG.ORBIT_BOX_WIDTH;
        targetHeight = CONFIG.ORBIT_BOX_HEIGHT;
      }
    } else {
      targetX = state.width / 2;
      targetY = state.height * 0.47;
      targetWidth = CONFIG.ORBIT_BOX_WIDTH;
      targetHeight = CONFIG.ORBIT_BOX_HEIGHT;
    }
  }

  state.boxX = lerp(
    state.boxX,
    targetX,
    CONFIG.BOX_SMOOTHING
  );

  state.boxY = lerp(
    state.boxY,
    targetY,
    CONFIG.BOX_SMOOTHING
  );

  state.boxWidth = lerp(
    state.boxWidth,
    targetWidth,
    CONFIG.BOX_SMOOTHING
  );

  state.boxHeight = lerp(
    state.boxHeight,
    targetHeight,
    CONFIG.BOX_SMOOTHING
  );
}

function createBoxPath(
  ctx,
  x,
  y,
  width,
  height,
  radius
) {
  const left = x - width / 2;
  const top = y - height / 2;

  const safeRadius = Math.min(
    radius,
    width / 2,
    height / 2
  );

  ctx.beginPath();
  ctx.roundRect(
    left,
    top,
    width,
    height,
    safeRadius
  );
}

function drawAbstractBlurField(
  ctx,
  x,
  y,
  width,
  height,
  opacity,
  now
) {
  const left = x - width / 2;
  const top = y - height / 2;

  ctx.save();

  createBoxPath(
    ctx,
    x,
    y,
    width,
    height,
    8
  );

  ctx.clip();

  ctx.filter =
    `grayscale(1) contrast(1.5) blur(` +
    `${CONFIG.PRIVACY_BLUR_AMOUNT}px)`;

  const blocks = 18;

  for (
    let block = 0;
    block < blocks;
    block++
  ) {
    const blockX =
      left +
      ((block * 71) % width) -
      width * 0.16 +
      Math.sin(
        now * 0.001 + block
      ) * 20;

    const blockY =
      top +
      ((block * 41) % height) -
      height * 0.14 +
      Math.cos(
        now * 0.0013 + block
      ) * 16;

    const blockWidth =
      width *
      randomRange(0.13, 0.34);

    const blockHeight =
      height *
      randomRange(0.2, 0.58);

    ctx.fillStyle =
      block % 2 === 0
        ? `rgba(190, 190, 190, ${
            0.18 * opacity
          })`
        : `rgba(26, 26, 26, ${
            0.23 * opacity
          })`;

    ctx.fillRect(
      blockX,
      blockY,
      blockWidth,
      blockHeight
    );
  }

  ctx.restore();

  drawBoxBoundary(
    ctx,
    x,
    y,
    width,
    height,
    opacity
  );
}

function drawCameraPrivacyBlur(
  ctx,
  x,
  y,
  width,
  height,
  now
) {
  if (
    !state.cameraStarted ||
    !video.videoWidth
  ) {
    drawAbstractBlurField(
      ctx,
      x,
      y,
      width,
      height,
      1,
      now
    );

    return;
  }

  const softness = CONFIG.EDGE_SOFTNESS;

  const paddedWidth = Math.ceil(
    width + softness * 2
  );

  const paddedHeight = Math.ceil(
    height + softness * 2
  );

  if (
    blurCanvas.width !== paddedWidth ||
    blurCanvas.height !== paddedHeight
  ) {
    blurCanvas.width = paddedWidth;
    blurCanvas.height = paddedHeight;
  }

  const screenLeft = x - width / 2;
  const screenTop = y - height / 2;

  const paddedScreenLeft =
    screenLeft - softness;

  const paddedScreenTop =
    screenTop - softness;

  const videoRect = getVideoCoverRect();

  let normalizedLeft =
    (paddedScreenLeft - videoRect.x) /
    videoRect.width;

  const normalizedTop =
    (paddedScreenTop - videoRect.y) /
    videoRect.height;

  const normalizedWidth =
    paddedWidth / videoRect.width;

  const normalizedHeight =
    paddedHeight / videoRect.height;

  if (CONFIG.MIRROR_CAMERA) {
    normalizedLeft =
      1 -
      normalizedLeft -
      normalizedWidth;
  }

  const sourceX =
    normalizedLeft * video.videoWidth;

  const sourceY =
    normalizedTop * video.videoHeight;

  const sourceWidth =
    normalizedWidth * video.videoWidth;

  const sourceHeight =
    normalizedHeight * video.videoHeight;

  blurCtx.clearRect(
    0,
    0,
    paddedWidth,
    paddedHeight
  );

  blurCtx.drawImage(
    video,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    paddedWidth,
    paddedHeight
  );

  ctx.save();

  createBoxPath(
    ctx,
    x,
    y,
    width,
    height,
    7
  );

  ctx.clip();

  ctx.filter = [
    `grayscale(${
      CONFIG.PRIVACY_GRAYSCALE
    })`,

    `contrast(${
      CONFIG.PRIVACY_CONTRAST
    })`,

    `brightness(${
      CONFIG.PRIVACY_BRIGHTNESS
    })`,

    `blur(${
      CONFIG.PRIVACY_BLUR_AMOUNT
    }px)`,
  ].join(" ");

  for (
    let pass = 0;
    pass < CONFIG.PRIVACY_BLUR_PASSES;
    pass++
  ) {
    ctx.globalAlpha =
      pass === 0 ? 1 : 0.52;

    ctx.drawImage(
      blurCanvas,
      paddedScreenLeft,
      paddedScreenTop,
      paddedWidth,
      paddedHeight
    );
  }

  ctx.restore();

  drawBoxBoundary(
    ctx,
    x,
    y,
    width,
    height,
    1
  );
}

function drawBoxBoundary(
  ctx,
  x,
  y,
  width,
  height,
  opacity
) {
  ctx.save();

  createBoxPath(
    ctx,
    x,
    y,
    width,
    height,
    7
  );

  ctx.lineWidth = 1;

  ctx.strokeStyle =
    `rgba(240, 240, 240, ${
      0.24 * opacity
    })`;

  ctx.shadowColor =
    `rgba(90, 90, 90, ${
      0.55 * opacity
    })`;

  ctx.shadowBlur = 12;

  ctx.stroke();
  ctx.restore();
}

function drawBlurBox(ctx, now) {
  drawCameraPrivacyBlur(
    ctx,
    state.boxX,
    state.boxY,
    state.boxWidth,
    state.boxHeight,
    now
  );
}

function addFogPuff(now) {
  const face = getFaceScreenData();

  const originX = face
    ? face.x
    : state.boxX;

  const originY = face
    ? face.y
    : state.boxY;

  const spread = face
    ? Math.max(
        Math.max(
          face.width,
          face.height
        ) * 1.7,
        190
      )
    : Math.min(
        state.width,
        state.height
      ) * 0.24;

  state.fogPuffs.push({
    x:
      originX +
      randomRange(-spread, spread),

    y:
      originY +
      randomRange(
        -spread * 0.56,
        spread * 0.56
      ),

    radius: randomRange(
      spread * 0.33,
      spread * 0.92
    ),

    alpha: randomRange(
      0.018,
      0.052
    ),

    hue: randomRange(
      202,
      239
    ),

    rotation: randomRange(
      0,
      Math.PI * 2
    ),

    createdAt: now,
  });

  if (
    state.fogPuffs.length >
    CONFIG.MAX_FOG_PUFFS
  ) {
    state.fogPuffs.shift();
  }
}

function updateFog(now) {
  if (
    getCurrentStage() !== 3 ||
    !isPhoneActive(now)
  ) {
    return;
  }

  if (
    now - state.lastFogSpawnAt >=
    CONFIG.FOG_SPAWN_INTERVAL_MS
  ) {
    addFogPuff(now);
    state.lastFogSpawnAt = now;
  }
}

function drawFog(ctx, now) {
  ctx.save();
  ctx.globalCompositeOperation = "screen";

  for (const puff of state.fogPuffs) {
    const ageSeconds =
      (now - puff.createdAt) / 1000;

    const x =
      puff.x +
      Math.sin(
        ageSeconds * 0.37 +
        puff.rotation
      ) * 11;

    const y =
      puff.y +
      Math.cos(
        ageSeconds * 0.31 +
        puff.rotation
      ) * 7;

    const gradient =
      ctx.createRadialGradient(
        x,
        y,
        puff.radius * 0.06,
        x,
        y,
        puff.radius
      );

    gradient.addColorStop(
      0,
      `hsla(${
        puff.hue
      }, 55%, 88%, ${
        puff.alpha
      })`
    );

    gradient.addColorStop(
      0.34,
      `hsla(${
        puff.hue
      }, 62%, 72%, ${
        puff.alpha * 0.72
      })`
    );

    gradient.addColorStop(
      0.72,
      `hsla(${
        puff.hue
      }, 72%, 51%, ${
        puff.alpha * 0.26
      })`
    );

    gradient.addColorStop(
      1,
      `hsla(${
        puff.hue
      }, 80%, 40%, 0)`
    );

    ctx.fillStyle = gradient;

    ctx.beginPath();

    ctx.ellipse(
      x,
      y,
      puff.radius,
      puff.radius * 0.7,
      puff.rotation,
      0,
      Math.PI * 2
    );

    ctx.fill();
  }

  ctx.restore();
}

function drawAmbientField(ctx, now) {
  const pulse =
    0.5 +
    Math.sin(now * 0.00081) *
    0.5;

  const radius =
    Math.max(
      state.width,
      state.height
    ) * 0.8;

  const field =
    ctx.createRadialGradient(
      state.width * 0.5,
      state.height * 0.46,
      0,
      state.width * 0.5,
      state.height * 0.46,
      radius
    );

  field.addColorStop(
    0,
    `rgba(38, 56, 122, ${
      0.052 + pulse * 0.025
    })`
  );

  field.addColorStop(
    0.55,
    "rgba(9, 10, 34, 0.025)"
  );

  field.addColorStop(
    1,
    "rgba(0, 0, 0, 0)"
  );

  ctx.fillStyle = field;

  ctx.fillRect(
    0,
    0,
    state.width,
    state.height
  );
}

function animationLoop(now) {
  requestAnimationFrame(animationLoop);

  visualCtx.clearRect(
    0,
    0,
    state.width,
    state.height
  );

  updatePersistentStage(now);
  updateStageUi(now);
  updateBlurBox(now);
  updateFog(now);

  drawAmbientField(
    visualCtx,
    now
  );

  drawBlurBox(
    visualCtx,
    now
  );

  drawFog(
    visualCtx,
    now
  );

  detect(now);
}

window.addEventListener(
  "resize",
  resizeCanvas
);

resizeCanvas();

/*
  The experience begins automatically.
  No intro screen or start-button click is required.
*/
beginExperience();

requestAnimationFrame(animationLoop);