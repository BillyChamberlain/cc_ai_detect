"use strict";

/* ==========================================================
   EDIT THESE FOR YOUR OWN MODELS
   ========================================================== */
const CONFIG = {
  FACE_MODEL_PATH: "./models/face_best.onnx",
  PHONE_MODEL_PATH: "./models/best.onnx",

  /*
    ADDED: InsightFace face-mapping model.
    For FaceAnalysis(name="buffalo_l"), this is commonly
    w600k_r50.onnx.
  */
  FACE_MAPPING_MODEL_PATH: "./models/w600k_r50.onnx",

  /*
    ADDED: enrollment data converted from me.npy.
  */
  ENROLLMENT_PATH: "./me.json",

  /*
    ADDED: image drawn over unknown faces.
  */
  FACE_COVER_PATH: "./cover.png",

  /*
    Must match the image size used while exporting YOLO to ONNX.
    Typical values: 320, 416, 512, or 640.
  */
  MODEL_INPUT_SIZE: 640,

  /*
    Face mapper input. InsightFace w600k_r50 is commonly 112x112.
  */
  FACE_MAPPING_INPUT_SIZE: 112,

  /*
    Use 0 if the face and phone models each contain only one class.
  */
  FACE_CLASS_ID: 0,
  PHONE_CLASS_ID: 0,

  FACE_CONFIDENCE: 0.45,
  PHONE_CONFIDENCE: 0.45,
  NMS_IOU_THRESHOLD: 0.45,

  /*
    Inference runs at a max of ~8 times per second.
  */
  INFERENCE_INTERVAL_MS: 125,

  /*
    Stage timeline:
    Stage 1: 0–15 sec, constructed blur only.
    Stage 2: at 15 sec, camera starts and face detection starts.
    Stage 3: first valid phone detection.
    Stage 4: fog delay after Stage 3 trigger.
  */
  CAMERA_STAGE_START_MS: 15_000,
  FOG_DELAY_AFTER_PHONE_MS: 20_000,

  /*
    The CSS video is mirrored.
    Inference and sampled blur coordinates are adjusted to match it.
  */
  MIRROR_CAMERA: true,

  FACE_HOLD_MS: 500,
  PHONE_HOLD_MS: 420,

  /*
    Stage 2:
    On first face appearance, box briefly matches face dimensions.
    It then shrinks to the normal size and follows a true circle.
  */
  FACE_LOCK_DURATION_MS: 700,
  ORBIT_BOX_WIDTH: 172,
  ORBIT_BOX_HEIGHT: 120,
  ORBIT_RADIUS_RATIO: 0.18,
  ORBIT_SPEED: 0.00105,
  ORBIT_Y_RATIO: 1,

  /*
    Stage 3 and Stage 4:
    Blur covers the face when the phone is visible.
  */
  FRONT_BOX_SCALE: 1.08,

  /*
    Movement smoothing.
  */
  BOX_SMOOTHING: 0.1,

  /*
    Stage 1 construction blur dimensions.
  */
  INITIAL_DRAW_DURATION_MS: 12_000,
  INITIAL_BOX_WIDTH_RATIO: 0.42,
  INITIAL_BOX_HEIGHT_RATIO: 0.19,

  /*
    Blur-box visual configuration.
    Only the moving rectangle gets this treatment.
  */
  PRIVACY_BLUR_AMOUNT: 36,
  PRIVACY_BLUR_PASSES: 2,
  PRIVACY_GRAYSCALE: 1,
  PRIVACY_CONTRAST: 1.8,
  PRIVACY_BRIGHTNESS: 0.82,
  EDGE_SOFTNESS: 22,

  /*
    Stage 4 fog.
  */
  FOG_SPAWN_INTERVAL_MS: 155,
  MAX_FOG_PUFFS: 250,

  /*
    ==========================================================
    ADDED: IDENTITY RECOGNITION
    ==========================================================
  */

  /*
    Set this to "cosine" if Python compared normalized
    InsightFace embeddings with cosine similarity.

    Set this to "euclidean" if best_match_score() uses
    Euclidean distance.
  */
  IDENTITY_METRIC: "cosine",

  /*
    Replace this with the threshold from your Python config.
  */
  IDENTITY_THRESHOLD: 0.52,

  /*
    Set true if your Python enrollment stored:
      face.normed_embedding

    Set false if it stored:
      face.embedding
    and did not normalize it afterward.
  */
  NORMALIZE_MAPPING_OUTPUT: true,

  /*
    If your Python mapper preprocessing used BGR instead of RGB,
    change this to "BGR".
  */
  MAPPING_CHANNEL_ORDER: "RGB",

  /*
    Common InsightFace preprocessing:
      (pixel / 127.5) - 1
  */
  MAPPING_NORMALIZATION: "minus-one-to-one",

  /*
    Temporal voting, equivalent to the Python version.
    Covers appear only after enough recent frames recognize you.
  */
  IDENTITY_VOTE_WINDOW: 8,
  IDENTITY_MATCHES_REQUIRED: 5,

  /*
    The identity overlay is active during Stage 2 and beyond.
    Set this false if you want it to begin at Stage 3 instead.
  */
  IDENTITY_OVERLAY_FROM_STAGE: 2,
};


/* ========================================================== */
/* DOM references                                             */
/* ========================================================== */

const video =
  document.getElementById("camera");

const visualCanvas =
  document.getElementById("visualCanvas");

const inferenceCanvas =
  document.getElementById("inferenceCanvas");

const visualCtx =
  visualCanvas.getContext("2d");

const inferenceCtx =
  inferenceCanvas.getContext(
    "2d",
    {
      willReadFrequently: true,
    }
  );

const blurCanvas =
  document.createElement("canvas");

const blurCtx =
  blurCanvas.getContext("2d");

const mapperCanvas =
  document.createElement("canvas");

const mapperCtx =
  mapperCanvas.getContext(
    "2d",
    {
      willReadFrequently: true,
    }
  );

/*
  FIX: reused scratch canvas for imageDataToTensor.
  The original code created a brand-new <canvas> element on every
  call to imageDataToTensor (i.e. up to ~8x/sec once a face was
  detected). That churn, combined with the missing re-entrancy
  guard below, was a major contributor to the memory growth that
  crashed the tab. Reusing a single off-DOM canvas avoids the
  per-frame allocation entirely.
*/
const mapperScratchCanvas =
  document.createElement("canvas");

const mapperScratchCtx =
  mapperScratchCanvas.getContext(
    "2d",
    {
      willReadFrequently: true,
    }
  );

const intro =
  document.getElementById("intro");

const startButton =
  document.getElementById("startButton");

const statusTitle =
  document.getElementById("statusTitle");

const statusText =
  document.getElementById("statusText");

const phaseLabel =
  document.getElementById("phaseLabel");

const modelLabel =
  document.getElementById("modelLabel");


/* ========================================================== */
/* State                                                       */
/* ========================================================== */

const state = {
  started: false,
  startTime: 0,

  cameraStarted: false,
  cameraStarting: false,

  faceModelLoading: false,
  phoneModelLoading: false,
  mappingModelLoading: false,

  faceSession: null,
  phoneSession: null,
  mappingSession: null,

  faceInputName: null,
  phoneInputName: null,
  mappingInputName: null,

  lastInferenceTime: 0,

  /*
    FIX: hard re-entrancy guard for detect().
    animationLoop() calls detect(now) on every requestAnimationFrame
    (roughly 60x/sec) without awaiting it. The original code only
    throttled how often a NEW inference cycle could *start*
    (INFERENCE_INTERVAL_MS), but did nothing to stop a new cycle
    from starting while a previous one (face model -> phone model ->
    mapping model, each with tensor allocation + canvas reads) was
    still in flight. On slower backends (e.g. the WASM fallback if
    WebGPU isn't available) a single cycle can take longer than
    125ms, so cycles stacked up indefinitely, each holding its own
    tensors/buffers, until the tab ran out of memory and crashed.
    This flag ensures only one inference cycle runs at a time.
  */
  isInferring: false,

  face: null,
  phone: null,

  lastFaceSeenAt: 0,
  lastPhoneSeenAt: 0,

  faceLockStartedAt: null,
  wasFaceActive: false,

  phoneTriggerTime: null,

  stage: 1,

  boxX: window.innerWidth / 2,
  boxY: window.innerHeight / 2,
  boxWidth: 260,
  boxHeight: 120,

  fogPuffs: [],
  lastFogSpawnAt: 0,

  width: window.innerWidth,
  height: window.innerHeight,

  /*
    ========================================================
    ADDED: identity state
    ========================================================
  */
  enrolledEmbeddings: [],
  coverImage: null,

  identityFaces: [],
  identityMatchHistory: [],

  currentIdentityPresent: false,
  stableIdentityPresent: false,

  lastIdentityError: null,
};


/* ========================================================== */
/* Basic utilities                                             */
/* ========================================================== */

function resizeCanvas() {
  const dpr =
    Math.min(
      window.devicePixelRatio || 1,
      2
    );

  state.width =
    window.innerWidth;

  state.height =
    window.innerHeight;

  visualCanvas.width =
    Math.floor(
      state.width * dpr
    );

  visualCanvas.height =
    Math.floor(
      state.height * dpr
    );

  visualCanvas.style.width =
    `${state.width}px`;

  visualCanvas.style.height =
    `${state.height}px`;

  visualCtx.setTransform(
    dpr,
    0,
    0,
    dpr,
    0,
    0
  );
}

function getElapsedMs() {
  if (!state.startTime) {
    return 0;
  }

  return performance.now() -
    state.startTime;
}

function clamp(value, min, max) {
  return Math.max(
    min,
    Math.min(max, value)
  );
}

function lerp(
  current,
  target,
  amount
) {
  return current +
    (target - current) *
    amount;
}

function randomRange(min, max) {
  return min +
    Math.random() *
    (max - min);
}

function easeOutCubic(value) {
  return 1 -
    Math.pow(
      1 - clamp(value, 0, 1),
      3
    );
}

function consoleOnce(key, ...values) {
  if (!consoleOnce.keys.has(key)) {
    consoleOnce.keys.add(key);
    console.log(...values);
  }
}

consoleOnce.keys =
  new Set();


/* ========================================================== */
/* Existing face/phone state                                   */
/* ========================================================== */

function isFaceActive(now) {
  return (
    state.face !== null &&
    now - state.lastFaceSeenAt <=
      CONFIG.FACE_HOLD_MS
  );
}

function isPhoneActive(now) {
  return (
    state.phone !== null &&
    now - state.lastPhoneSeenAt <=
      CONFIG.PHONE_HOLD_MS
  );
}


/* ========================================================== */
/* Existing persistent stage logic                             */
/* ========================================================== */

function updatePersistentStage(now) {
  if (!state.started) {
    state.stage = 1;
    return;
  }

  const elapsed =
    getElapsedMs();

  if (
    state.stage === 1 &&
    elapsed >=
      CONFIG.CAMERA_STAGE_START_MS
  ) {
    state.stage = 2;
  }

  if (
    state.stage < 3 &&
    state.phoneTriggerTime !== null
  ) {
    state.stage = 3;
  }

  if (
    state.stage === 3 &&
    state.phoneTriggerTime !== null &&
    now -
      state.phoneTriggerTime >=
      CONFIG.FOG_DELAY_AFTER_PHONE_MS
  ) {
    state.stage = 4;
  }
}

function getCurrentStage() {
  return state.stage;
}


/* ========================================================== */
/* Existing camera coordinate mapping                          */
/* ========================================================== */

function getVideoCoverRect() {
  const videoWidth =
    video.videoWidth || 1;

  const videoHeight =
    video.videoHeight || 1;

  const screenAspect =
    state.width /
    state.height;

  const videoAspect =
    videoWidth /
    videoHeight;

  let width;
  let height;
  let x;
  let y;

  if (
    videoAspect >
    screenAspect
  ) {
    height =
      state.height;

    width =
      height *
      videoAspect;

    x =
      (state.width - width) /
      2;

    y =
      0;
  } else {
    width =
      state.width;

    height =
      width /
      videoAspect;

    x =
      0;

    y =
      (state.height - height) /
      2;
  }

  return {
    x,
    y,
    width,
    height,
  };
}

function mapModelPointToScreen(
  modelX,
  modelY
) {
  const videoRect =
    getVideoCoverRect();

  const normalizedX =
    modelX /
    CONFIG.MODEL_INPUT_SIZE;

  const normalizedY =
    modelY /
    CONFIG.MODEL_INPUT_SIZE;

  return {
    x:
      videoRect.x +
      normalizedX *
      videoRect.width,

    y:
      videoRect.y +
      normalizedY *
      videoRect.height,
  };
}

function getFaceScreenData() {
  if (!state.face) {
    return null;
  }

  const center =
    mapModelPointToScreen(
      state.face.x,
      state.face.y
    );

  const videoRect =
    getVideoCoverRect();

  const screenFaceWidth =
    state.face.width /
    CONFIG.MODEL_INPUT_SIZE *
    videoRect.width;

  const screenFaceHeight =
    state.face.height /
    CONFIG.MODEL_INPUT_SIZE *
    videoRect.height;

  return {
    x: center.x,
    y: center.y,
    width: Math.max(
      56,
      screenFaceWidth
    ),
    height: Math.max(
      56,
      screenFaceHeight
    ),
  };
}


/* ========================================================== */
/* ONNX sessions                                               */
/* ========================================================== */

async function createOrtSession(
  modelPath
) {
  try {
    return await ort.InferenceSession.create(
      modelPath,
      {
        executionProviders: [
          "webgpu",
        ],
        graphOptimizationLevel: "all",
      }
    );
  } catch (webGpuError) {
    console.warn(
      "WebGPU unavailable. Falling back to WASM.",
      webGpuError
    );

    return ort.InferenceSession.create(
      modelPath,
      {
        executionProviders: [
          "wasm",
        ],
        graphOptimizationLevel: "all",
      }
    );
  }
}

async function loadFaceModel() {
  if (
    state.faceSession ||
    state.faceModelLoading
  ) {
    return;
  }

  state.faceModelLoading =
    true;

  try {
    state.faceSession =
      await createOrtSession(
        CONFIG.FACE_MODEL_PATH
      );

    state.faceInputName =
      state.faceSession.inputNames[0];

    consoleOnce(
      "face-model-info",
      "Face detector inputs:",
      state.faceSession.inputNames,
      "outputs:",
      state.faceSession.outputNames
    );
  } finally {
    state.faceModelLoading =
      false;
  }
}

async function loadPhoneModel() {
  if (
    state.phoneSession ||
    state.phoneModelLoading
  ) {
    return;
  }

  state.phoneModelLoading =
    true;

  try {
    state.phoneSession =
      await createOrtSession(
        CONFIG.PHONE_MODEL_PATH
      );

    state.phoneInputName =
      state.phoneSession.inputNames[0];

    consoleOnce(
      "phone-model-info",
      "Phone detector inputs:",
      state.phoneSession.inputNames,
      "outputs:",
      state.phoneSession.outputNames
    );
  } finally {
    state.phoneModelLoading =
      false;
  }
}


/*
  ADDED: load InsightFace mapping model.
*/
async function loadMappingModel() {
  if (
    state.mappingSession ||
    state.mappingModelLoading
  ) {
    return;
  }

  state.mappingModelLoading =
    true;

  try {
    state.mappingSession =
      await createOrtSession(
        CONFIG.FACE_MAPPING_MODEL_PATH
      );

    state.mappingInputName =
      state.mappingSession.inputNames[0];

    consoleOnce(
      "mapping-model-info",
      "Face mapper inputs:",
      state.mappingSession.inputNames,
      "outputs:",
      state.mappingSession.outputNames
    );
  } finally {
    state.mappingModelLoading =
      false;
  }
}


/* ========================================================== */
/* Enrollment and overlay loading                              */
/* ========================================================== */

async function loadEnrollmentEmbeddings() {
  const response =
    await fetch(
      CONFIG.ENROLLMENT_PATH
    );

  if (!response.ok) {
    throw new Error(
      `Could not load ${CONFIG.ENROLLMENT_PATH}`
    );
  }

  const json =
    await response.json();

  const rawEmbeddings =
    Array.isArray(json)
      ? json
      : json.embeddings;

  if (
    !Array.isArray(rawEmbeddings)
  ) {
    throw new Error(
      "me.json must contain an embeddings array."
    );
  }

  const embeddings =
    rawEmbeddings.map(
      values =>
        Float32Array.from(values)
    );

  if (
    embeddings.length === 0
  ) {
    throw new Error(
      "me.json contains no embeddings."
    );
  }

  const expectedLength =
    embeddings[0].length;

  for (
    const embedding of embeddings
  ) {
    if (
      embedding.length !==
      expectedLength
    ) {
      throw new Error(
        "Enrollment embeddings have inconsistent lengths."
      );
    }
  }

  console.log(
    "Loaded enrolled embeddings:",
    embeddings.length,
    "vectors of length:",
    expectedLength
  );

  return embeddings;
}

function loadImage(path) {
  return new Promise(
    (resolve, reject) => {
      const image =
        new Image();

      image.onload =
        () => resolve(image);

      image.onerror =
        () => reject(
          new Error(
            `Could not load image: ${path}`
          )
        );

      image.src =
        path;
    }
  );
}


/* ========================================================== */
/* Camera                                                       */
/* ========================================================== */

async function startCamera() {
  if (
    state.cameraStarted ||
    state.cameraStarting
  ) {
    return;
  }

  if (
    !navigator.mediaDevices ||
    !navigator.mediaDevices.getUserMedia
  ) {
    throw new Error(
      "Camera access requires localhost or HTTPS."
    );
  }

  state.cameraStarting =
    true;

  try {
    const stream =
      await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: {
            ideal: 1280,
          },
          height: {
            ideal: 720,
          },
        },
      });

    video.srcObject =
      stream;

    await new Promise(
      resolve => {
        video.onloadedmetadata =
          resolve;
      }
    );

    await video.play();

    state.cameraStarted =
      true;

    video.classList.add(
      "visible"
    );
  } finally {
    state.cameraStarting =
      false;
  }
}

async function beginStageTwo() {
  if (
    state.cameraStarted ||
    state.cameraStarting
  ) {
    return;
  }

  try {
    modelLabel.textContent =
      "Opening camera";

    await Promise.all([
      startCamera(),
      loadFaceModel(),
      loadPhoneModel(),
      loadMappingModel(),
    ]);

    modelLabel.textContent =
      "Searching for face";
  } catch (error) {
    console.error(
      "Stage 2 initialization failed:",
      error
    );

    modelLabel.textContent =
      "Camera or model error";
  }
}


/* ========================================================== */
/* Existing camera-to-inference image                           */
/* ========================================================== */

function drawVideoToInferenceCanvas() {
  const size =
    CONFIG.MODEL_INPUT_SIZE;

  inferenceCanvas.width =
    size;

  inferenceCanvas.height =
    size;

  inferenceCtx.clearRect(
    0,
    0,
    size,
    size
  );

  inferenceCtx.fillStyle =
    "#000";

  inferenceCtx.fillRect(
    0,
    0,
    size,
    size
  );

  const videoWidth =
    video.videoWidth;

  const videoHeight =
    video.videoHeight;

  if (
    !videoWidth ||
    !videoHeight
  ) {
    return;
  }

  const scale =
    Math.min(
      size / videoWidth,
      size / videoHeight
    );

  const drawWidth =
    videoWidth * scale;

  const drawHeight =
    videoHeight * scale;

  const offsetX =
    (size - drawWidth) / 2;

  const offsetY =
    (size - drawHeight) / 2;

  if (
    CONFIG.MIRROR_CAMERA
  ) {
    inferenceCtx.save();

    inferenceCtx.translate(
      size,
      0
    );

    inferenceCtx.scale(
      -1,
      1
    );

    inferenceCtx.drawImage(
      video,
      offsetX,
      offsetY,
      drawWidth,
      drawHeight
    );

    inferenceCtx.restore();

    return;
  }

  inferenceCtx.drawImage(
    video,
    offsetX,
    offsetY,
    drawWidth,
    drawHeight
  );
}

function createInputTensor() {
  const size =
    CONFIG.MODEL_INPUT_SIZE;

  const imageData =
    inferenceCtx.getImageData(
      0,
      0,
      size,
      size
    );

  const pixels =
    imageData.data;

  const channelSize =
    size * size;

  const input =
    new Float32Array(
      3 * channelSize
    );

  for (
    let pixelIndex = 0;
    pixelIndex < channelSize;
    pixelIndex++
  ) {
    const sourceIndex =
      pixelIndex * 4;

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
    [
      1,
      3,
      size,
      size,
    ]
  );
}


/* ========================================================== */
/* Existing YOLO decoding                                      */
/* ========================================================== */

function calculateIoU(
  first,
  second
) {
  const firstLeft =
    first.x -
    first.width / 2;

  const firstTop =
    first.y -
    first.height / 2;

  const firstRight =
    first.x +
    first.width / 2;

  const firstBottom =
    first.y +
    first.height / 2;

  const secondLeft =
    second.x -
    second.width / 2;

  const secondTop =
    second.y -
    second.height / 2;

  const secondRight =
    second.x +
    second.width / 2;

  const secondBottom =
    second.y +
    second.height / 2;

  const overlapLeft =
    Math.max(
      firstLeft,
      secondLeft
    );

  const overlapTop =
    Math.max(
      firstTop,
      secondTop
    );

  const overlapRight =
    Math.min(
      firstRight,
      secondRight
    );

  const overlapBottom =
    Math.min(
      firstBottom,
      secondBottom
    );

  const overlapWidth =
    Math.max(
      0,
      overlapRight -
        overlapLeft
    );

  const overlapHeight =
    Math.max(
      0,
      overlapBottom -
        overlapTop
    );

  const overlapArea =
    overlapWidth *
    overlapHeight;

  const firstArea =
    (
      firstRight -
      firstLeft
    ) *
    (
      firstBottom -
      firstTop
    );

  const secondArea =
    (
      secondRight -
      secondLeft
    ) *
    (
      secondBottom -
      secondTop
    );

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

function nonMaximumSuppression(
  detections
) {
  const candidates =
    [...detections].sort(
      (a, b) =>
        b.confidence -
        a.confidence
    );

  const selected =
    [];

  while (
    candidates.length > 0
  ) {
    const best =
      candidates.shift();

    selected.push(
      best
    );

    for (
      let index =
        candidates.length - 1;
      index >= 0;
      index--
    ) {
      if (
        calculateIoU(
          best,
          candidates[index]
        ) >
        CONFIG.NMS_IOU_THRESHOLD
      ) {
        candidates.splice(
          index,
          1
        );
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
  const {
    data,
    dims,
  } =
    outputTensor;

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
    const detections =
      [];

    const count =
      dims[1];

    for (
      let index = 0;
      index < count;
      index++
    ) {
      const offset =
        index * 6;

      const x1 =
        data[offset];

      const y1 =
        data[offset + 1];

      const x2 =
        data[offset + 2];

      const y2 =
        data[offset + 3];

      const confidence =
        data[offset + 4];

      const classId =
        Math.round(
          data[offset + 5]
        );

      if (
        classId === wantedClassId &&
        confidence >= confidenceThreshold
      ) {
        detections.push({
          x:
            (x1 + x2) / 2,

          y:
            (y1 + y2) / 2,

          width:
            x2 - x1,

          height:
            y2 - y1,

          confidence,
        });
      }
    }

    return nonMaximumSuppression(
      detections
    )[0] || null;
  }

  /*
    Raw YOLO output:
    [1, 4 + classCount, anchorCount]
    or:
    [1, anchorCount, 4 + classCount]
  */
  if (
    dims.length !== 3
  ) {
    console.warn(
      "Unexpected YOLO output dimensions:",
      dims
    );

    return null;
  }

  const attributesFirst =
    dims[1] < dims[2];

  const attributes =
    attributesFirst
      ? dims[1]
      : dims[2];

  const anchors =
    attributesFirst
      ? dims[2]
      : dims[1];

  if (
    attributes < 5
  ) {
    console.warn(
      "Invalid YOLO output shape:",
      dims
    );

    return null;
  }

  const detections =
    [];

  for (
    let anchor = 0;
    anchor < anchors;
    anchor++
  ) {
    const read =
      attribute => {
        if (
          attributesFirst
        ) {
          return data[
            attribute *
              anchors +
              anchor
          ];
        }

        return data[
          anchor *
            attributes +
            attribute
        ];
      };

    const x =
      read(0);

    const y =
      read(1);

    const width =
      read(2);

    const height =
      read(3);

    let highestScore =
      -Infinity;

    let detectedClassId =
      -1;

    for (
      let attribute = 4;
      attribute < attributes;
      attribute++
    ) {
      const score =
        read(attribute);

      if (
        score > highestScore
      ) {
        highestScore =
          score;

        detectedClassId =
          attribute - 4;
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
        confidence:
          highestScore,
      });
    }
  }

  return nonMaximumSuppression(
    detections
  )[0] || null;
}

async function runModel(
  session,
  inputName,
  wantedClassId,
  confidenceThreshold,
  inputTensor
) {
  const outputs =
    await session.run({
      [inputName]:
        inputTensor,
    });

  const outputTensor =
    outputs[
      session.outputNames[0]
    ];

  return parseYoloOutput(
    outputTensor,
    wantedClassId,
    confidenceThreshold
  );
}


/* ========================================================== */
/* Existing detector                                           */
/* ========================================================== */

async function detect(now) {
  if (
    !state.started ||
    !state.cameraStarted
  ) {
    return;
  }

  if (
    !state.faceSession ||
    !video.videoWidth
  ) {
    return;
  }

  /*
    FIX: bail out immediately if a previous inference cycle
    (started on an earlier animation frame) hasn't finished yet.
    Without this, overlapping async cycles could pile up and
    exhaust memory. See the comment on state.isInferring above.
  */
  if (
    state.isInferring
  ) {
    return;
  }

  if (
    now - state.lastInferenceTime <
    CONFIG.INFERENCE_INTERVAL_MS
  ) {
    return;
  }

  state.isInferring =
    true;

  state.lastInferenceTime =
    now;

  try {
    drawVideoToInferenceCanvas();

    const inputTensor =
      createInputTensor();

    const faceDetection =
      await runModel(
        state.faceSession,
        state.faceInputName,
        CONFIG.FACE_CLASS_ID,
        CONFIG.FACE_CONFIDENCE,
        inputTensor
      );

    if (
      faceDetection
    ) {
      state.face =
        faceDetection;

      state.lastFaceSeenAt =
        now;
    }

    if (
      state.phoneSession
    ) {
      const phoneDetection =
        await runModel(
          state.phoneSession,
          state.phoneInputName,
          CONFIG.PHONE_CLASS_ID,
          CONFIG.PHONE_CONFIDENCE,
          inputTensor
        );

      if (
        phoneDetection
      ) {
        state.phone =
          phoneDetection;

        state.lastPhoneSeenAt =
          now;

        if (
          state.phoneTriggerTime ===
          null
        ) {
          state.phoneTriggerTime =
            now;

          state.stage =
            3;
        }
      }
    }

    /*
      ADDED:
      Identity recognition runs on the existing detected
      face box. It does not replace the detector.
    */
    if (
      state.faceSession &&
      state.mappingSession
    ) {
      await recognizeDetectedFace(
        now
      );
    }
  } catch (error) {
    console.error(
      "Inference error:",
      error
    );

    state.lastIdentityError =
      error;

    modelLabel.textContent =
      "Inference paused";
  } finally {
    /*
      FIX: always release the lock, whether the cycle succeeded
      or threw, so a single failure can't wedge inference off
      forever, and so the next frame's detect() call is allowed
      to start a fresh cycle.
    */
    state.isInferring =
      false;
  }
}


/* ========================================================== */
/* ADDED: identity recognition                                 */
/* ========================================================== */

async function recognizeDetectedFace(
  now
) {
  /*
    This version uses the face detector's current box.

    Your current detector returns one face because the existing
    code uses [0]. To cover multiple people, the detector must
    return all face detections. See the note after the code.
  */
  if (
    !state.face
  ) {
    state.identityFaces =
      [];

    updateIdentityVoting(
      false
    );

    return;
  }

  const faceBox =
    detectionToPixelBox(
      state.face
    );

  const faceCrop =
    cropFrameBox(
      inferenceCanvas,
      faceBox
    );

  if (
    !faceCrop
  ) {
    updateIdentityVoting(
      false
    );

    return;
  }

  const embedding =
    await createMappingEmbedding(
      faceCrop
    );

  const match =
    findBestIdentityMatch(
      embedding,
      state.enrolledEmbeddings
    );

  const isMe =
    classifyIdentityMatch(
      match
    );

  state.identityFaces =
    [
      {
        detection:
          state.face,

        box:
          state.face,

        screenBox:
          getFaceScreenBox(
            state.face
          ),

        match,
        isMe,
      },
    ];

  updateIdentityVoting(
    isMe
  );
}

function detectionToPixelBox(
  detection
) {
  /*
    Your face detector uses model coordinates.
    Convert its center-format box from the 640x640
    inference image into pixel coordinates on the
    inference canvas.
  */
  return {
    x:
      detection.x -
      detection.width / 2,

    y:
      detection.y -
      detection.height / 2,

    width:
      detection.width,

    height:
      detection.height,
  };
}

function cropFrameBox(
  canvas,
  box
) {
  const x1 =
    clamp(
      Math.floor(box.x),
      0,
      canvas.width
    );

  const y1 =
    clamp(
      Math.floor(box.y),
      0,
      canvas.height
    );

  const x2 =
    clamp(
      Math.ceil(
        box.x +
        box.width
      ),
      0,
      canvas.width
    );

  const y2 =
    clamp(
      Math.ceil(
        box.y +
        box.height
      ),
      0,
      canvas.height
    );

  const width =
    x2 - x1;

  const height =
    y2 - y1;

  if (
    width <= 2 ||
    height <= 2
  ) {
    return null;
  }

  return inferenceCtx.getImageData(
    x1,
    y1,
    width,
    height
  );
}

async function createMappingEmbedding(
  faceCrop
) {
  const tensorInfo =
    imageDataToTensor(
      faceCrop,
      CONFIG.FACE_MAPPING_INPUT_SIZE,
      CONFIG.FACE_MAPPING_INPUT_SIZE,
      {
        channelOrder:
          CONFIG.MAPPING_CHANNEL_ORDER,

        normalization:
          CONFIG.MAPPING_NORMALIZATION,

        layout:
          "NCHW",
      }
    );

  const inputTensor =
    new ort.Tensor(
      "float32",
      tensorInfo.data,
      tensorInfo.shape
    );

  const outputs =
    await state.mappingSession.run({
      [state.mappingInputName]:
        inputTensor,
    });

  const outputName =
    state.mappingSession.outputNames[0];

  const output =
    outputs[outputName];

  if (
    !output
  ) {
    throw new Error(
      "The face-mapping output was not found."
    );
  }

  let embedding =
    Float32Array.from(
      output.data
    );

  const expectedLength =
    state.enrolledEmbeddings[0].length;

  if (
    embedding.length !==
    expectedLength
  ) {
    throw new Error(
      `Mapper output has ${embedding.length} values, ` +
      `but me.json has ${expectedLength}.`
    );
  }

  if (
    CONFIG.NORMALIZE_MAPPING_OUTPUT
  ) {
    embedding =
      l2Normalize(
        embedding
      );
  }

  return embedding;
}

function imageDataToTensor(
  imageData,
  targetWidth,
  targetHeight,
  options
) {
  mapperCanvas.width =
    targetWidth;

  mapperCanvas.height =
    targetHeight;

  mapperCtx.clearRect(
    0,
    0,
    targetWidth,
    targetHeight
  );

  /*
    FIX: reuse a single off-DOM scratch canvas instead of calling
    document.createElement("canvas") on every invocation. The
    original allocated and discarded a brand-new canvas element
    up to ~8x/sec once a face was detected, adding GC pressure on
    top of the concurrent-inference issue fixed in detect().
  */
  mapperScratchCanvas.width =
    imageData.width;

  mapperScratchCanvas.height =
    imageData.height;

  mapperScratchCtx.putImageData(
    imageData,
    0,
    0
  );

  /*
    This resizes the detector crop directly.

    If your Python FaceAnalysis pipeline aligns faces
    using landmarks before recognition, replace this
    with the same alignment step.
  */
  mapperCtx.drawImage(
    mapperScratchCanvas,
    0,
    0,
    targetWidth,
    targetHeight
  );

  const pixels =
    mapperCtx.getImageData(
      0,
      0,
      targetWidth,
      targetHeight
    ).data;

  const area =
    targetWidth *
    targetHeight;

  const tensor =
    new Float32Array(
      area * 3
    );

  for (
    let y = 0;
    y < targetHeight;
    y++
  ) {
    for (
      let x = 0;
      x < targetWidth;
      x++
    ) {
      const pixelIndex =
        (
          y *
          targetWidth +
          x
        ) * 4;

      let red =
        pixels[pixelIndex];

      let green =
        pixels[pixelIndex + 1];

      let blue =
        pixels[pixelIndex + 2];

      if (
        options.channelOrder ===
        "BGR"
      ) {
        const temporary =
          red;

        red =
          blue;

        blue =
          temporary;
      }

      red =
        normalizePixel(
          red,
          options.normalization
        );

      green =
        normalizePixel(
          green,
          options.normalization
        );

      blue =
        normalizePixel(
          blue,
          options.normalization
        );

      const index =
        y *
        targetWidth +
        x;

      /*
        NCHW layout.
      */
      tensor[index] =
        red;

      tensor[area + index] =
        green;

      tensor[2 * area + index] =
        blue;
    }
  }

  return {
    data:
      tensor,

    shape: [
      1,
      3,
      targetHeight,
      targetWidth,
    ],
  };
}

function normalizePixel(
  pixel,
  mode
) {
  if (
    mode ===
    "none"
  ) {
    return pixel;
  }

  if (
    mode ===
    "zero-to-one"
  ) {
    return pixel / 255;
  }

  if (
    mode ===
    "minus-one-to-one"
  ) {
    return (
      pixel / 127.5
    ) - 1;
  }

  throw new Error(
    `Unknown normalization mode: ${mode}`
  );
}

function findBestIdentityMatch(
  embedding,
  enrolled
) {
  if (
    CONFIG.IDENTITY_METRIC ===
    "cosine"
  ) {
    let bestSimilarity =
      -Infinity;

    let bestIndex =
      -1;

    for (
      let index = 0;
      index < enrolled.length;
      index++
    ) {
      const similarity =
        cosineSimilarity(
          embedding,
          enrolled[index]
        );

      if (
        similarity >
        bestSimilarity
      ) {
        bestSimilarity =
          similarity;

        bestIndex =
          index;
      }
    }

    return {
      metric:
        "cosine",

      value:
        bestSimilarity,

      index:
        bestIndex,
    };
  }

  let bestDistance =
    Infinity;

  let bestIndex =
    -1;

  for (
    let index = 0;
    index < enrolled.length;
    index++
  ) {
    const distance =
      euclideanDistance(
        embedding,
        enrolled[index]
      );

    if (
      distance <
      bestDistance
    ) {
      bestDistance =
        distance;

      bestIndex =
        index;
    }
  }

  return {
    metric:
      "euclidean",

    value:
      bestDistance,

    index:
      bestIndex,
  };
}

function classifyIdentityMatch(
  match
) {
  if (
    match.metric ===
    "cosine"
  ) {
    return (
      match.value >=
      CONFIG.IDENTITY_THRESHOLD
    );
  }

  return (
    match.value <=
    CONFIG.IDENTITY_THRESHOLD
  );
}

function updateIdentityVoting(
  foundMe
) {
  state.currentIdentityPresent =
    foundMe;

  state.identityMatchHistory.push(
    foundMe
  );

  if (
    state.identityMatchHistory.length >
    CONFIG.IDENTITY_VOTE_WINDOW
  ) {
    state.identityMatchHistory.shift();
  }

  state.stableIdentityPresent =
    state.identityMatchHistory.filter(
      Boolean
    ).length >=
    CONFIG.IDENTITY_MATCHES_REQUIRED;
}

function euclideanDistance(
  first,
  second
) {
  if (
    first.length !==
    second.length
  ) {
    throw new Error(
      "Embedding lengths do not match."
    );
  }

  let total =
    0;

  for (
    let index = 0;
    index < first.length;
    index++
  ) {
    const difference =
      first[index] -
      second[index];

    total +=
      difference *
      difference;
  }

  return Math.sqrt(
    total
  );
}

function cosineSimilarity(
  first,
  second
) {
  if (
    first.length !==
    second.length
  ) {
    throw new Error(
      "Embedding lengths do not match."
    );
  }

  let dot =
    0;

  let normFirst =
    0;

  let normSecond =
    0;

  for (
    let index = 0;
    index < first.length;
    index++
  ) {
    dot +=
      first[index] *
      second[index];

    normFirst +=
      first[index] *
      first[index];

    normSecond +=
      second[index] *
      second[index];
  }

  if (
    normFirst === 0 ||
    normSecond === 0
  ) {
    return -1;
  }

  return (
    dot /
    (
      Math.sqrt(normFirst) *
      Math.sqrt(normSecond)
    )
  );
}

function l2Normalize(
  values
) {
  let total =
    0;

  for (
    const value of values
  ) {
    total +=
      value *
      value;
  }

  const magnitude =
    Math.sqrt(total);

  if (
    magnitude === 0
  ) {
    return values;
  }

  return Float32Array.from(
    values,
    value =>
      value / magnitude
  );
}


/* ========================================================== */
/* ADDED: screen box and overlay                                */
/* ========================================================== */

function getFaceScreenBox(
  detection
) {
  const videoRect =
    getVideoCoverRect();

  const left =
    detection.x -
    detection.width / 2;

  const top =
    detection.y -
    detection.height / 2;

  const right =
    detection.x +
    detection.width / 2;

  const bottom =
    detection.y +
    detection.height / 2;

  const topLeft =
    mapModelPointToScreen(
      left,
      top
    );

  const bottomRight =
    mapModelPointToScreen(
      right,
      bottom
    );

  return {
    x:
      topLeft.x,

    y:
      topLeft.y,

    width:
      bottomRight.x -
      topLeft.x,

    height:
      bottomRight.y -
      topLeft.y,
  };
}

function drawIdentityOverlays(
  ctx
) {
  if (
    getCurrentStage() <
    CONFIG.IDENTITY_OVERLAY_FROM_STAGE
  ) {
    return;
  }

  if (
    !state.stableIdentityPresent
  ) {
    return;
  }

  for (
    const face of state.identityFaces
  ) {
    if (
      face.isMe
    ) {
      continue;
    }

    drawCoverCenteredOnBox(
      ctx,
      state.coverImage,
      face.screenBox
    );
  }
}

function drawCoverCenteredOnBox(
  ctx,
  image,
  box
) {
  if (
    !image ||
    !box
  ) {
    return;
  }

  const centerX =
    box.x +
    box.width / 2;

  const centerY =
    box.y +
    box.height / 2;

  const x =
    Math.round(
      centerX -
      image.width / 2
    );

  const y =
    Math.round(
      centerY -
      image.height / 2
    );

  /*
    No resizing occurs here.
    The image keeps its original dimensions.
  */
  ctx.drawImage(
    image,
    x,
    y,
    image.width,
    image.height
  );
}


/* ========================================================== */
/* Existing blur-box logic                                     */
/* ========================================================== */

function updateBlurBox(now) {
  const elapsed =
    getElapsedMs();

  const stage =
    getCurrentStage();

  let targetX =
    state.width / 2;

  let targetY =
    state.height / 2;

  let targetWidth =
    state.width *
    CONFIG.INITIAL_BOX_WIDTH_RATIO;

  let targetHeight =
    state.height *
    CONFIG.INITIAL_BOX_HEIGHT_RATIO;

  if (
    stage === 1
  ) {
    targetX =
      state.width / 2 +
      Math.sin(
        now * 0.00032
      ) *
      state.width *
      0.017;

    targetY =
      state.height / 2 +
      Math.cos(
        now * 0.00028
      ) *
      state.height *
      0.014;
  }

  const faceActive =
    isFaceActive(now);

  if (
    faceActive &&
    !state.wasFaceActive
  ) {
    state.faceLockStartedAt =
      now;
  }

  if (
    !faceActive
  ) {
    state.faceLockStartedAt =
      null;
  }

  state.wasFaceActive =
    faceActive;

  if (
    stage === 2
  ) {
    if (
      faceActive
    ) {
      const face =
        getFaceScreenData();

      const faceLockAge =
        state.faceLockStartedAt ===
        null
          ? Infinity
          : now -
            state.faceLockStartedAt;

      const isInFaceLock =
        faceLockAge <
        CONFIG.FACE_LOCK_DURATION_MS;

      if (
        isInFaceLock
      ) {
        targetX =
          face.x;

        targetY =
          face.y;

        targetWidth =
          face.width *
          1.15;

        targetHeight =
          face.height *
          1.18;
      } else {
        const orbitRadius =
          Math.max(
            Math.max(
              face.width,
              face.height
            ) * 1.45,

            Math.min(
              state.width,
              state.height
            ) *
            CONFIG.ORBIT_RADIUS_RATIO
          );

        const orbitAngle =
          elapsed *
          CONFIG.ORBIT_SPEED;

        targetX =
          face.x +
          Math.cos(
            orbitAngle
          ) *
          orbitRadius;

        targetY =
          face.y +
          Math.sin(
            orbitAngle
          ) *
          orbitRadius *
          CONFIG.ORBIT_Y_RATIO;

        targetWidth =
          CONFIG.ORBIT_BOX_WIDTH;

        targetHeight =
          CONFIG.ORBIT_BOX_HEIGHT;
      }
    } else {
      targetX =
        state.width *
        0.5 +
        Math.cos(
          elapsed *
          0.00047
        ) *
        state.width *
        0.12;

      targetY =
        state.height *
        0.46 +
        Math.sin(
          elapsed *
          0.00062
        ) *
        44;

      targetWidth =
        CONFIG.ORBIT_BOX_WIDTH;

      targetHeight =
        CONFIG.ORBIT_BOX_HEIGHT;
    }
  }

  if (
    stage === 3 ||
    stage === 4
  ) {
    if (
      faceActive
    ) {
      const face =
        getFaceScreenData();

      if (
        isPhoneActive(now)
      ) {
        targetX =
          face.x;

        targetY =
          face.y;

        targetWidth =
          face.width *
          CONFIG.FRONT_BOX_SCALE;

        targetHeight =
          face.height *
          CONFIG.FRONT_BOX_SCALE;
      } else {
        const orbitRadius =
          Math.max(
            Math.max(
              face.width,
              face.height
            ) * 1.45,

            Math.min(
              state.width,
              state.height
            ) *
            CONFIG.ORBIT_RADIUS_RATIO
          );

        const orbitAngle =
          elapsed *
          CONFIG.ORBIT_SPEED;

        targetX =
          face.x +
          Math.cos(
            orbitAngle
          ) *
          orbitRadius;

        targetY =
          face.y +
          Math.sin(
            orbitAngle
          ) *
          orbitRadius *
          CONFIG.ORBIT_Y_RATIO;

        targetWidth =
          CONFIG.ORBIT_BOX_WIDTH;

        targetHeight =
          CONFIG.ORBIT_BOX_HEIGHT;
      }
    } else {
      targetX =
        state.width / 2;

      targetY =
        state.height *
        0.47;

      targetWidth =
        CONFIG.ORBIT_BOX_WIDTH;

      targetHeight =
        CONFIG.ORBIT_BOX_HEIGHT;
    }
  }

  state.boxX =
    lerp(
      state.boxX,
      targetX,
      CONFIG.BOX_SMOOTHING
    );

  state.boxY =
    lerp(
      state.boxY,
      targetY,
      CONFIG.BOX_SMOOTHING
    );

  state.boxWidth =
    lerp(
      state.boxWidth,
      targetWidth,
      CONFIG.BOX_SMOOTHING
    );

  state.boxHeight =
    lerp(
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
  const left =
    x -
    width / 2;

  const top =
    y -
    height / 2;

  const safeRadius =
    Math.min(
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

function drawInitialConstruction(
  ctx,
  now
) {
  const elapsed =
    getElapsedMs();

  const progress =
    easeOutCubic(
      elapsed /
      CONFIG.INITIAL_DRAW_DURATION_MS
    );

  const width =
    state.width *
    CONFIG.INITIAL_BOX_WIDTH_RATIO;

  const height =
    state.height *
    CONFIG.INITIAL_BOX_HEIGHT_RATIO;

  const x =
    state.boxX;

  const y =
    state.boxY;

  const left =
    x -
    width / 2;

  const top =
    y -
    height / 2;

  const totalLines =
    62;

  const visibleLines =
    Math.floor(
      totalLines *
      progress
    );

  ctx.save();

  ctx.filter =
    `blur(${CONFIG.PRIVACY_BLUR_AMOUNT}px)`;

  for (
    let line = 0;
    line < visibleLines;
    line++
  ) {
    const lineProgress =
      line /
      totalLines;

    const lineY =
      top +
      lineProgress *
      height;

    const unstableStart =
      left +
      Math.sin(
        line *
        1.72 +
        now *
        0.006
      ) *
      (
        1 -
        progress
      ) *
      width *
      0.12;

    const lineWidth =
      width *
      (
        0.72 +
        Math.sin(
          line *
          1.27 +
          now *
          0.003
        ) *
        0.12
      );

    ctx.globalAlpha =
      0.18 +
      Math.sin(
        line *
        0.9 +
        now *
        0.004
      ) *
      0.04;

    ctx.fillStyle =
      line % 3 === 0
        ? "rgba(190, 212, 255, 0.82)"
        : "rgba(112, 126, 167, 0.9)";

    ctx.fillRect(
      unstableStart,
      lineY,
      lineWidth,
      Math.max(
        3,
        (
          height /
          totalLines
        ) *
        1.65
      )
    );
  }

  ctx.restore();

  ctx.save();

  const perimeter =
    2 *
    (
      width +
      height
    );

  const drawnPerimeter =
    perimeter *
    progress;

  ctx.lineWidth =
    1.2;

  ctx.strokeStyle =
    "rgba(214, 226, 255, 0.56)";

  ctx.shadowColor =
    "rgba(119, 154, 255, 0.95)";

  ctx.shadowBlur =
    15;

  ctx.beginPath();

  let remaining =
    drawnPerimeter;

  if (
    remaining > 0
  ) {
    const topLength =
      Math.min(
        width,
        remaining
      );

    ctx.moveTo(
      left,
      top
    );

    ctx.lineTo(
      left +
      topLength,
      top
    );

    remaining -=
      topLength;
  }

  if (
    remaining > 0
  ) {
    const rightLength =
      Math.min(
        height,
        remaining
      );

    ctx.moveTo(
      left +
      width,
      top
    );

    ctx.lineTo(
      left +
      width,
      top +
      rightLength
    );

    remaining -=
      rightLength;
  }

  if (
    remaining > 0
  ) {
    const bottomLength =
      Math.min(
        width,
        remaining
      );

    ctx.moveTo(
      left +
      width,
      top +
      height
    );

    ctx.lineTo(
      left +
      width -
      bottomLength,
      top +
      height
    );

    remaining -=
      bottomLength;
  }

  if (
    remaining > 0
  ) {
    const leftLength =
      Math.min(
        height,
        remaining
      );

    ctx.moveTo(
      left,
      top +
      height
    );

    ctx.lineTo(
      left,
      top +
      height -
      leftLength
    );
  }

  ctx.stroke();

  ctx.restore();

  if (
    progress > 0.72
  ) {
    const settledOpacity =
      (
        progress -
        0.72
      ) /
      0.28;

    drawAbstractBlurField(
      ctx,
      x,
      y,
      width,
      height,
      settledOpacity *
      0.78,
      now
    );
  }
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
  const left =
    x -
    width / 2;

  const top =
    y -
    height / 2;

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
    "grayscale(1) contrast(1.5) blur(36px)";

  const blocks =
    18;

  for (
    let block = 0;
    block < blocks;
    block++
  ) {
    const blockX =
      left +
      (
        block *
        71
      ) %
      width -
      width *
      0.16 +
      Math.sin(
        now *
        0.001 +
        block
      ) *
      20;

    const blockY =
      top +
      (
        block *
        41
      ) %
      height -
      height *
      0.14 +
      Math.cos(
        now *
        0.0013 +
        block
      ) *
      16;

    const blockWidth =
      width *
      randomRange(
        0.13,
        0.34
      );

    const blockHeight =
      height *
      randomRange(
        0.2,
        0.58
      );

    ctx.fillStyle =
      block % 2 === 0
        ? `rgba(190, 190, 190, ${0.18 * opacity})`
        : `rgba(26, 26, 26, ${0.23 * opacity})`;

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

  const softness =
    CONFIG.EDGE_SOFTNESS;

  const paddedWidth =
    Math.ceil(
      width +
      softness *
      2
    );

  const paddedHeight =
    Math.ceil(
      height +
      softness *
      2
    );

  if (
    blurCanvas.width !==
    paddedWidth ||
    blurCanvas.height !==
    paddedHeight
  ) {
    blurCanvas.width =
      paddedWidth;

    blurCanvas.height =
      paddedHeight;
  }

  const screenLeft =
    x -
    width / 2;

  const screenTop =
    y -
    height / 2;

  const paddedScreenLeft =
    screenLeft -
    softness;

  const paddedScreenTop =
    screenTop -
    softness;

  const videoRect =
    getVideoCoverRect();

  let normalizedLeft =
    (
      paddedScreenLeft -
      videoRect.x
    ) /
    videoRect.width;

  const normalizedTop =
    (
      paddedScreenTop -
      videoRect.y
    ) /
    videoRect.height;

  const normalizedWidth =
    paddedWidth /
    videoRect.width;

  const normalizedHeight =
    paddedHeight /
    videoRect.height;

  if (
    CONFIG.MIRROR_CAMERA
  ) {
    normalizedLeft =
      1 -
      normalizedLeft -
      normalizedWidth;
  }

  const sourceX =
    normalizedLeft *
    video.videoWidth;

  const sourceY =
    normalizedTop *
    video.videoHeight;

  const sourceWidth =
    normalizedWidth *
    video.videoWidth;

  const sourceHeight =
    normalizedHeight *
    video.videoHeight;

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
    `grayscale(${CONFIG.PRIVACY_GRAYSCALE})`,
    `contrast(${CONFIG.PRIVACY_CONTRAST})`,
    `brightness(${CONFIG.PRIVACY_BRIGHTNESS})`,
    `blur(${CONFIG.PRIVACY_BLUR_AMOUNT}px)`,
  ].join(" ");

  for (
    let pass = 0;
    pass <
    CONFIG.PRIVACY_BLUR_PASSES;
    pass++
  ) {
    ctx.globalAlpha =
      pass === 0
        ? 1
        : 0.52;

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

  ctx.lineWidth =
    1;

  ctx.strokeStyle =
    `rgba(240, 240, 240, ${0.24 * opacity})`;

  ctx.shadowColor =
    `rgba(90, 90, 90, ${0.55 * opacity})`;

  ctx.shadowBlur =
    12;

  ctx.stroke();

  ctx.restore();
}

function drawBlurBox(
  ctx,
  now
) {
  const stage =
    getCurrentStage();

  if (
    stage === 1
  ) {
    drawInitialConstruction(
      ctx,
      now
    );

    return;
  }

  drawCameraPrivacyBlur(
    ctx,
    state.boxX,
    state.boxY,
    state.boxWidth,
    state.boxHeight,
    now
  );
}


/* ========================================================== */
/* Existing fog                                               */
/* ========================================================== */

function addFogPuff(now) {
  const face =
    getFaceScreenData();

  const originX =
    face
      ? face.x
      : state.boxX;

  const originY =
    face
      ? face.y
      : state.boxY;

  const spread =
    face
      ? Math.max(
          Math.max(
            face.width,
            face.height
          ) *
          1.7,
          190
        )
      : Math.min(
          state.width,
          state.height
        ) *
        0.24;

  state.fogPuffs.push({
    x:
      originX +
      randomRange(
        -spread,
        spread
      ),

    y:
      originY +
      randomRange(
        -spread *
        0.56,
        spread *
        0.56
      ),

    radius:
      randomRange(
        spread *
        0.33,
        spread *
        0.92
      ),

    alpha:
      randomRange(
        0.018,
        0.052
      ),

    hue:
      randomRange(
        202,
        239
      ),

    rotation:
      randomRange(
        0,
        Math.PI * 2
      ),

    createdAt:
      now,
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
    getCurrentStage() !== 4 ||
    !isPhoneActive(now)
  ) {
    return;
  }

  if (
    now -
    state.lastFogSpawnAt >=
    CONFIG.FOG_SPAWN_INTERVAL_MS
  ) {
    addFogPuff(now);

    state.lastFogSpawnAt =
      now;
  }
}

function drawFog(
  ctx,
  now
) {
  ctx.save();

  ctx.globalCompositeOperation =
    "screen";

  for (
    const puff of state.fogPuffs
  ) {
    const ageSeconds =
      (
        now -
        puff.createdAt
      ) /
      1000;

    const x =
      puff.x +
      Math.sin(
        ageSeconds *
        0.37 +
        puff.rotation
      ) *
      11;

    const y =
      puff.y +
      Math.cos(
        ageSeconds *
        0.31 +
        puff.rotation
      ) *
      7;

    const gradient =
      ctx.createRadialGradient(
        x,
        y,
        puff.radius *
        0.06,
        x,
        y,
        puff.radius
      );

    gradient.addColorStop(
      0,
      `hsla(${puff.hue}, 55%, 88%, ${puff.alpha})`
    );

    gradient.addColorStop(
      0.34,
      `hsla(${puff.hue}, 62%, 72%, ${puff.alpha * 0.72})`
    );

    gradient.addColorStop(
      0.72,
      `hsla(${puff.hue}, 72%, 51%, ${puff.alpha * 0.26})`
    );

    gradient.addColorStop(
      1,
      `hsla(${puff.hue}, 80%, 40%, 0)`
    );

    ctx.fillStyle =
      gradient;

    ctx.beginPath();

    ctx.ellipse(
      x,
      y,
      puff.radius,
      puff.radius *
      0.7,
      puff.rotation,
      0,
      Math.PI * 2
    );

    ctx.fill();
  }

  ctx.restore();
}


/* ========================================================== */
/* Existing ambient field                                      */
/* ========================================================== */

function drawAmbientField(
  ctx,
  now
) {
  const pulse =
    0.5 +
    Math.sin(
      now *
      0.00081
    ) *
    0.5;

  const radius =
    Math.max(
      state.width,
      state.height
    ) *
    0.8;

  const field =
    ctx.createRadialGradient(
      state.width *
      0.5,
      state.height *
      0.46,
      0,
      state.width *
      0.5,
      state.height *
      0.46,
      radius
    );

  field.addColorStop(
    0,
    `rgba(38, 56, 122, ${0.052 + pulse * 0.025})`
  );

  field.addColorStop(
    0.55,
    "rgba(9, 10, 34, 0.025)"
  );

  field.addColorStop(
    1,
    "rgba(0, 0, 0, 0)"
  );

  ctx.fillStyle =
    field;

  ctx.fillRect(
    0,
    0,
    state.width,
    state.height
  );
}


/* ========================================================== */
/* Existing UI                                                 */
/* ========================================================== */

function updateStageUi(now) {
  const stage =
    getCurrentStage();

  if (
    stage === 1
  ) {
    phaseLabel.textContent =
      "Stage 01 · Drawing blur";

    modelLabel.textContent =
      "Camera offline";

    return;
  }

  if (
    stage === 2
  ) {
    phaseLabel.textContent =
      "Stage 02 · Circular orbit: The threat of lurks, try bringing a phone into frame";

    if (
      state.cameraStarting
    ) {
      modelLabel.textContent =
        "Opening camera";
    } else if (
      !state.cameraStarted
    ) {
      modelLabel.textContent =
        "Camera waiting";
    } else if (
      state.stableIdentityPresent
    ) {
      modelLabel.textContent =
        "You recognized · other faces covered";
    } else if (
      isFaceActive(now)
    ) {
      modelLabel.textContent =
        "Face detected · identifying";
    } else {
      modelLabel.textContent =
        "Searching for face";
    }

    return;
  }

  if (
    stage === 3
  ) {
    const timeRemaining =
      Math.max(
        0,
        CONFIG.FOG_DELAY_AFTER_PHONE_MS -
          (
            now -
            state.phoneTriggerTime
          )
      );

    const secondsRemaining =
      Math.ceil(
        timeRemaining /
        1000
      );

    phaseLabel.textContent =
      "Stage 03 · Phone response, blur covers face";

    modelLabel.textContent =
      isPhoneActive(now)
        ? `Phone active · fog in ${secondsRemaining}s`
        : `Phone response entered · fog in ${secondsRemaining}s`;

    return;
  }

  phaseLabel.textContent =
    "Stage 04 · Permanent fog everytime phone is brought";

  modelLabel.textContent =
    isPhoneActive(now)
      ? "Phone active · fog accumulating"
      : "Stage 04 active · fog remains";
}


/* ========================================================== */
/* Existing animation loop                                    */
/* ========================================================== */

function animationLoop(now) {
  requestAnimationFrame(
    animationLoop
  );

  visualCtx.clearRect(
    0,
    0,
    state.width,
    state.height
  );

  updatePersistentStage(
    now
  );

  if (
    state.started &&
    getCurrentStage() >= 2
  ) {
    beginStageTwo();
  }

  updatePersistentStage(
    now
  );

  updateStageUi(
    now
  );

  updateBlurBox(
    now
  );

  updateFog(
    now
  );

  drawAmbientField(
    visualCtx,
    now
  );

  drawBlurBox(
    visualCtx,
    now
  );

  /*
    ADDED:
    This is drawn after the original blur/fog content.
    Therefore the cover appears above the existing effects.
  */
  drawIdentityOverlays(
    visualCtx
  );

  drawFog(
    visualCtx,
    now
  );

  detect(
    now
  );
}


/* ========================================================= */
/* Existing experience start                                  */
/* ========================================================= */

function beginExperience() {
  if (
    state.started
  ) {
    return;
  }

  state.started =
    true;

  state.startTime =
    performance.now();

  startButton.disabled =
    true;

  intro.classList.add(
    "hidden"
  );

  phaseLabel.textContent =
    "Stage 01 · Drawing blur";

  modelLabel.textContent =
    "Camera offline";
}

startButton.addEventListener(
  "click",
  beginExperience
);

window.addEventListener(
  "resize",
  resizeCanvas
);


/* ========================================================== */
/* Startup                                                     */
/* ========================================================== */

async function loadInitialAssets() {
  try {
    state.enrolledEmbeddings =
      await loadEnrollmentEmbeddings();

    state.coverImage =
      await loadImage(
        CONFIG.FACE_COVER_PATH
      );
  } catch (error) {
    console.error(
      "Could not load identity assets:",
      error
    );

    state.lastIdentityError =
      error;

    modelLabel.textContent =
      "Identity assets unavailable";
  }
}

resizeCanvas();

loadInitialAssets();

requestAnimationFrame(
  animationLoop
);