import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import GUI from "lil-gui";
import { ActivationNotation, ParameterNotation } from "./ParameterNotation";

export interface SceneParameter {
  id: string;
  kind: "weight" | "bias";
  layer: number;
  source: number | null;
  destination: number;
  value: number;
}

interface SceneState {
  colors: THREE.BufferAttribute;
  opacities: THREE.BufferAttribute;
  edgeIds: string[];
  layerMeshes: THREE.InstancedMesh[];
  neuronBaseColors: THREE.Color[][];
}

interface HoverPreview {
  id: string;
  x: number;
  y: number;
  alignRight: boolean;
}

export function TrainingNetworkScene({
  architecture,
  activations,
  parameters,
  selectedId,
  onSelect,
}: {
  architecture: readonly number[];
  activations: number[][];
  parameters: SceneParameter[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneStateRef = useRef<SceneState | null>(null);
  const onSelectRef = useRef(onSelect);
  const [hovered, setHovered] = useState<HoverPreview | null>(null);
  const parameterById = useMemo(
    () => new Map(parameters.map((parameter) => [parameter.id, parameter])),
    [parameters],
  );
  const parameterByIdRef = useRef(parameterById);
  const activationsRef = useRef(activations);
  const hoveredParameter = hovered ? parameterById.get(hovered.id) : undefined;
  const hoveredInputIndex =
    hovered?.id.startsWith("a-0-") ? Number(hovered.id.slice(4)) : null;

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    parameterByIdRef.current = parameterById;
  }, [parameterById]);

  useEffect(() => {
    activationsRef.current = activations;
  }, [activations]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);

    const initialCameraPosition = new THREE.Vector3(22, 10, 40);
    const initialTarget = new THREE.Vector3(0, 0, 0);
    const orthographicViewHeight = 20;
    const perspectiveCamera = new THREE.PerspectiveCamera(42, 1, 0.1, 120);
    const orthographicCamera = new THREE.OrthographicCamera(
      -10,
      10,
      10,
      -10,
      0.1,
      120,
    );
    perspectiveCamera.position.copy(initialCameraPosition);
    orthographicCamera.position.copy(initialCameraPosition);
    let activeCamera: THREE.PerspectiveCamera | THREE.OrthographicCamera =
      orthographicCamera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.cursor = "grab";
    renderer.domElement.style.touchAction = "none";
    mount.appendChild(renderer.domElement);

    const createControls = (
      camera: THREE.PerspectiveCamera | THREE.OrthographicCamera,
      target: THREE.Vector3,
    ) => {
      const nextControls = new OrbitControls(camera, renderer.domElement);
      nextControls.enableDamping = true;
      nextControls.target.copy(target);
      nextControls.minDistance = 8;
      nextControls.maxDistance = 55;
      nextControls.minZoom = 0.35;
      nextControls.maxZoom = 4;
      nextControls.update();
      return nextControls;
    };

    let controls = createControls(activeCamera, initialTarget);
    const initialDistance = controls.getDistance();
    orthographicCamera.zoom =
      orthographicViewHeight /
      (2 *
        initialDistance *
        Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov / 2)));
    orthographicCamera.updateProjectionMatrix();

    const cameraSettings = {
      orthographic: true,
      distance: initialDistance,
      fieldOfView: perspectiveCamera.fov,
      orthographicZoom: orthographicCamera.zoom,
      resetCamera: () => {
        activeCamera.position.copy(initialCameraPosition);
        controls.target.copy(initialTarget);
        perspectiveCamera.fov = 42;
        orthographicCamera.zoom =
          orthographicViewHeight /
          (2 *
            initialDistance *
            Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov / 2)));
        perspectiveCamera.updateProjectionMatrix();
        orthographicCamera.updateProjectionMatrix();
        cameraSettings.distance = initialDistance;
        cameraSettings.fieldOfView = perspectiveCamera.fov;
        cameraSettings.orthographicZoom = orthographicCamera.zoom;
        distanceController.updateDisplay();
        fieldOfViewController.updateDisplay();
        orthographicZoomController.updateDisplay();
        controls.update();
      },
    };

    const gui = new GUI({ container: mount, title: "Camera", width: 245 });
    gui.domElement.style.position = "absolute";
    gui.domElement.style.top = "12px";
    gui.domElement.style.right = "12px";
    gui.domElement.style.zIndex = "10";
    gui.domElement.style.setProperty("--background-color", "#161b22");
    gui.domElement.style.setProperty("--title-background-color", "#21262d");
    gui.domElement.style.setProperty("--widget-color", "#30363d");
    gui.domElement.style.setProperty("--highlight-color", "#484f58");
    gui.domElement.style.setProperty("--text-color", "#e6edf3");
    gui.domElement.style.setProperty("--number-color", "#58a6ff");

    const distanceController = gui
      .add(cameraSettings, "distance", 8, 55, 0.1)
      .name("Distance (depth)")
      .onChange((distance: number) => {
        const direction = activeCamera.position
          .clone()
          .sub(controls.target)
          .normalize();
        activeCamera.position
          .copy(controls.target)
          .addScaledVector(direction, distance);
        activeCamera.updateProjectionMatrix();
        controls.update();
      });
    const fieldOfViewController = gui
      .add(cameraSettings, "fieldOfView", 20, 90, 1)
      .name("Field of view")
      .onChange((fieldOfView: number) => {
        perspectiveCamera.fov = fieldOfView;
        perspectiveCamera.updateProjectionMatrix();
      })
      .hide();
    const orthographicZoomController = gui
      .add(cameraSettings, "orthographicZoom", 0.35, 4, 0.01)
      .name("Zoom")
      .onChange((zoom: number) => {
        orthographicCamera.zoom = zoom;
        orthographicCamera.updateProjectionMatrix();
      });

    const syncCameraControls = () => {
      cameraSettings.distance = controls.getDistance();
      cameraSettings.orthographicZoom = orthographicCamera.zoom;
      distanceController.updateDisplay();
      orthographicZoomController.updateDisplay();
    };

    const configureControls = (target: THREE.Vector3) => {
      controls = createControls(activeCamera, target);
      controls.addEventListener("change", syncCameraControls);
    };
    controls.addEventListener("change", syncCameraControls);

    gui
      .add(cameraSettings, "orthographic")
      .name("Orthographic")
      .onChange((useOrthographic: boolean) => {
        const previousCamera = activeCamera;
        const target = controls.target.clone();
        const position = previousCamera.position.clone();
        const quaternion = previousCamera.quaternion.clone();
        const direction = position.clone().sub(target).normalize();
        controls.dispose();

        if (useOrthographic) {
          const distance = position.distanceTo(target);
          orthographicCamera.zoom =
            orthographicViewHeight /
            (2 *
              distance *
              Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov / 2)));
          cameraSettings.orthographicZoom = orthographicCamera.zoom;
        } else {
          const distance =
            orthographicViewHeight /
            (2 *
              orthographicCamera.zoom *
              Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov / 2)));
          position.copy(target).addScaledVector(direction, distance);
          cameraSettings.distance = distance;
        }

        activeCamera = useOrthographic ? orthographicCamera : perspectiveCamera;
        activeCamera.position.copy(position);
        activeCamera.quaternion.copy(quaternion);
        activeCamera.updateProjectionMatrix();
        configureControls(target);

        fieldOfViewController.show(!useOrthographic);
        orthographicZoomController.show(useOrthographic);
        distanceController.name(useOrthographic ? "Distance (depth)" : "Distance");
        syncCameraControls();
      });
    gui.add(cameraSettings, "resetCamera").name("Reset camera");

    scene.add(new THREE.AmbientLight(0xffffff, 1.7));
    const light = new THREE.DirectionalLight(0xffffff, 2.4);
    light.position.set(12, 18, 20);
    scene.add(light);

    const layerPositions = architecture.map((count, layer) =>
      createLayerPositions(count, layer, architecture.length),
    );
    const layerMeshes: THREE.InstancedMesh[] = [];
    const layerGeometries: THREE.BufferGeometry[] = [];
    const layerMaterials: THREE.Material[] = [];
    const neuronBaseColors: THREE.Color[][] = [];
    const transform = new THREE.Object3D();

    const neuronSize = 0.72;

    architecture.forEach((count, layer) => {
      const isInput = layer === 0;
      const geometry = new THREE.BoxGeometry(neuronSize, neuronSize, neuronSize);
      const material = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        metalness: isInput ? 0.08 : 0.22,
        roughness: isInput ? 0.72 : 0.46,
      });
      const mesh = new THREE.InstancedMesh(geometry, material, count);
      const inactive = new THREE.Color(isInput ? 0x242b34 : 0x28313b);
      const active = new THREE.Color(0xf0f6fc);
      const baseColors: THREE.Color[] = [];

      layerPositions[layer].forEach((position, index) => {
        transform.position.copy(position);
        transform.updateMatrix();
        mesh.setMatrixAt(index, transform.matrix);
        const activation = activationsRef.current[layer]?.[index] ?? 0;
        const color = inactive.clone().lerp(active, activation);
        mesh.setColorAt(index, color);
        baseColors.push(color);
      });

      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      scene.add(mesh);
      layerMeshes.push(mesh);
      layerGeometries.push(geometry);
      layerMaterials.push(material);
      neuronBaseColors.push(baseColors);
    });

    const weightParameters: SceneParameter[] = [];
    for (let layer = 0; layer < architecture.length - 1; layer++) {
      for (
        let destination = 0;
        destination < architecture[layer + 1];
        destination++
      ) {
        for (let source = 0; source < architecture[layer]; source++) {
          const id = `w-${layer}-${destination}-${source}`;
          weightParameters.push(
            parameterByIdRef.current.get(id) ?? {
              id,
              kind: "weight",
              layer,
              source,
              destination,
              value: 0,
            },
          );
        }
      }
    }
    const positions = new Float32Array(weightParameters.length * 6);
    const colors = new Float32Array(weightParameters.length * 6);
    const opacities = new Float32Array(weightParameters.length * 2);
    const positive = new THREE.Color(0x58a6ff);
    const negative = new THREE.Color(0xff6b63);
    const edgeIds: string[] = [];

    weightParameters.forEach((parameter, index) => {
      const destination = layerPositions[parameter.layer + 1][parameter.destination];
      const source = layerPositions[parameter.layer][parameter.source ?? 0];
      const color = parameter.value >= 0 ? positive : negative;
      const opacity = 0.025 + Math.min(Math.abs(parameter.value) / 0.6, 1) * 0.2;
      const positionOffset = index * 6;
      const opacityOffset = index * 2;

      positions.set(
        [source.x, source.y, source.z, destination.x, destination.y, destination.z],
        positionOffset,
      );
      colors.set([color.r, color.g, color.b, color.r, color.g, color.b], positionOffset);
      opacities[opacityOffset] = opacity;
      opacities[opacityOffset + 1] = opacity;
      edgeIds.push(parameter.id);
    });

    const connectionGeometry = new THREE.BufferGeometry();
    connectionGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const colorAttribute = new THREE.BufferAttribute(colors, 3);
    const opacityAttribute = new THREE.BufferAttribute(opacities, 1);
    colorAttribute.setUsage(THREE.DynamicDrawUsage);
    opacityAttribute.setUsage(THREE.DynamicDrawUsage);
    connectionGeometry.setAttribute("color", colorAttribute);
    connectionGeometry.setAttribute("connectionOpacity", opacityAttribute);

    const connectionMaterial = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      vertexShader: `
        attribute vec3 color;
        attribute float connectionOpacity;
        varying vec3 vColor;
        varying float vOpacity;

        void main() {
          vColor = color;
          vOpacity = connectionOpacity;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vOpacity;

        void main() {
          gl_FragColor = vec4(vColor, vOpacity);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }
      `,
    });
    const lines = new THREE.LineSegments(connectionGeometry, connectionMaterial);
    scene.add(lines);

    sceneStateRef.current = {
      colors: colorAttribute,
      opacities: opacityAttribute,
      edgeIds,
      layerMeshes,
      neuronBaseColors,
    };

    const raycaster = new THREE.Raycaster();
    raycaster.params.Line = { threshold: 0.55 };
    const pointer = new THREE.Vector2();
    const getSceneItemAtPointer = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, activeCamera);

      const neuronHit = raycaster.intersectObjects(layerMeshes, false)[0];
      if (neuronHit?.instanceId !== undefined) {
        const layer = layerMeshes.indexOf(neuronHit.object as THREE.InstancedMesh);
        if (layer === 0) return `a-0-${neuronHit.instanceId}`;
        return `b-${layer - 1}-${neuronHit.instanceId}`;
      }

      const hit = raycaster.intersectObject(lines, false)[0];
      if (!hit || hit.index === undefined) return null;
      return edgeIds[Math.floor(hit.index / 2)] ?? null;
    };

    const handlePointerMove = (event: PointerEvent) => {
      const id = getSceneItemAtPointer(event);
      const rect = renderer.domElement.getBoundingClientRect();
      renderer.domElement.style.cursor = id
        ? parameterByIdRef.current.has(id)
          ? "pointer"
          : "crosshair"
        : "grab";
      setHovered(
        id
          ? {
              id,
              x: event.clientX - rect.left + 12,
              y: event.clientY - rect.top + 12,
              alignRight: event.clientX - rect.left > rect.width - 150,
            }
          : null,
      );
    };
    const handleClick = (event: PointerEvent) => {
      const id = getSceneItemAtPointer(event);
      if (id && parameterByIdRef.current.has(id)) onSelectRef.current(id);
    };
    const handlePointerLeave = () => {
      renderer.domElement.style.cursor = "grab";
      setHovered(null);
    };
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("click", handleClick);
    renderer.domElement.addEventListener("pointerleave", handlePointerLeave);

    const resizeObserver = new ResizeObserver(() => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      const aspect = width / height;
      perspectiveCamera.aspect = aspect;
      perspectiveCamera.updateProjectionMatrix();
      orthographicCamera.left = (-orthographicViewHeight * aspect) / 2;
      orthographicCamera.right = (orthographicViewHeight * aspect) / 2;
      orthographicCamera.top = orthographicViewHeight / 2;
      orthographicCamera.bottom = -orthographicViewHeight / 2;
      orthographicCamera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    });
    resizeObserver.observe(mount);

    let frame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, activeCamera);
      frame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("click", handleClick);
      renderer.domElement.removeEventListener("pointerleave", handlePointerLeave);
      controls.dispose();
      gui.destroy();
      connectionGeometry.dispose();
      connectionMaterial.dispose();
      layerGeometries.forEach((geometry) => geometry.dispose());
      layerMaterials.forEach((material) => material.dispose());
      renderer.dispose();
      renderer.domElement.remove();
      layerMeshes.length = 0;
      sceneStateRef.current = null;
    };
  }, [architecture]);

  useEffect(() => {
    const state = sceneStateRef.current;
    if (!state) return;
    const colorArray = state.colors.array as Float32Array;
    const opacityArray = state.opacities.array as Float32Array;
    const positive = new THREE.Color(0x58a6ff);
    const negative = new THREE.Color(0xff6b63);

    state.edgeIds.forEach((id, index) => {
      const colorOffset = index * 6;
      const opacityOffset = index * 2;
      const value = parameterById.get(id)?.value ?? 0;
      const baseColor = value >= 0 ? positive : negative;
      const baseOpacity = 0.025 + Math.min(Math.abs(value) / 0.6, 1) * 0.2;
      const selected = id === selectedId;
      const previewed = id === hovered?.id;

      for (let vertex = 0; vertex < 2; vertex++) {
        const vertexOffset = colorOffset + vertex * 3;
        const red = selected
          ? 1
          : previewed
            ? baseColor.r + (1 - baseColor.r) * 0.58
            : baseColor.r;
        const green = selected
          ? 1
          : previewed
            ? baseColor.g + (1 - baseColor.g) * 0.58
            : baseColor.g;
        const blue = selected
          ? 1
          : previewed
            ? baseColor.b + (1 - baseColor.b) * 0.58
            : baseColor.b;
        colorArray[vertexOffset] = red;
        colorArray[vertexOffset + 1] = green;
        colorArray[vertexOffset + 2] = blue;
      }
      const opacity = selected
        ? 1
        : previewed
          ? 0.82
          : baseOpacity * 0.38;
      opacityArray[opacityOffset] = opacity;
      opacityArray[opacityOffset + 1] = opacity;
    });

    const selectedBias = selectedId.match(/^b-(\d+)-(\d+)$/);
    const selectedBiasLayer = selectedBias ? Number(selectedBias[1]) + 1 : -1;
    const selectedBiasNeuron = selectedBias ? Number(selectedBias[2]) : -1;
    const hoveredBias = hovered?.id.match(/^b-(\d+)-(\d+)$/);
    const hoveredInput = hovered?.id.match(/^a-0-(\d+)$/);
    const hoveredNeuronLayer = hoveredBias ? Number(hoveredBias[1]) + 1 : 0;
    const hoveredNeuron = hoveredBias
      ? Number(hoveredBias[2])
      : hoveredInput
        ? Number(hoveredInput[1])
        : -1;
    const selectedBiasColor = new THREE.Color(0xd29922);
    const hoveredNeuronColor = new THREE.Color(0x58a6ff);
    const activeNeuronColor = new THREE.Color(0xf0f6fc);

    state.layerMeshes.forEach((mesh, layer) => {
      const inactiveNeuronColor = new THREE.Color(
        layer === 0 ? 0x242b34 : 0x28313b,
      );
      state.neuronBaseColors[layer].forEach((baseColor, neuron) => {
        const activation = Math.max(
          0,
          Math.min(1, activations[layer]?.[neuron] ?? 0),
        );
        baseColor
          .copy(inactiveNeuronColor)
          .lerp(activeNeuronColor, activation);
        const selected = layer === selectedBiasLayer && neuron === selectedBiasNeuron;
        const previewed = layer === hoveredNeuronLayer && neuron === hoveredNeuron;
        mesh.setColorAt(
          neuron,
          selected ? selectedBiasColor : previewed ? hoveredNeuronColor : baseColor,
        );
      });
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    });

    state.colors.needsUpdate = true;
    state.opacities.needsUpdate = true;
  }, [activations, hovered?.id, parameterById, selectedId]);

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        minHeight: "560px",
        overflow: "hidden",
        borderRadius: "10px",
        background: "#0d1117",
      }}
    >
      <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />
      {hovered && (hoveredParameter || hoveredInputIndex !== null) && (
        <div
          style={{
            position: "absolute",
            left: `${hovered.x}px`,
            top: `${hovered.y}px`,
            zIndex: 20,
            minWidth: "126px",
            padding: "9px 11px",
            border: "1px solid #484f58",
            borderRadius: "7px",
            background: "rgba(13,17,23,0.96)",
            boxShadow: "0 8px 24px rgba(0,0,0,0.34)",
            color: "#f0f6fc",
            pointerEvents: "none",
            transform: hovered.alignRight ? "translateX(-100%)" : undefined,
          }}
        >
          {hoveredParameter ? (
            <ParameterNotation parameter={hoveredParameter} fontSize="17px" />
          ) : (
            <ActivationNotation index={hoveredInputIndex ?? 0} fontSize="17px" />
          )}
          <p
            style={{
              margin: "5px 0 0",
              color: "#8b949e",
              fontSize: "10px",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {hoveredParameter
              ? `${hoveredParameter.kind === "weight" ? "Weight" : "Bias"} · ${formatSceneValue(hoveredParameter.value)}`
              : `Input activation · ${formatSceneValue(activations[0]?.[hoveredInputIndex ?? 0] ?? 0)}`}
          </p>
        </div>
      )}
      <p
        style={{
          position: "absolute",
          left: "16px",
          bottom: "14px",
          color: "#6e7681",
          fontSize: "11px",
          pointerEvents: "none",
        }}
      >
        Hover to preview · Click to select · Drag to orbit
      </p>
    </div>
  );
}

function createLayerPositions(count: number, layer: number, layerCount: number) {
  const x = layerX(layer, layerCount);

  if (layer > 0) {
    const spacing = 0.88;
    return Array.from(
      { length: count },
      (_, index) => new THREE.Vector3(x, ((count - 1) / 2 - index) * spacing, 0),
    );
  }

  const side = 16;
  const rows = Math.ceil(count / side);
  const spacing = 0.88;

  return Array.from({ length: count }, (_, index) => {
    const column = index % side;
    const row = Math.floor(index / side);
    return new THREE.Vector3(
      x,
      ((rows - 1) / 2 - row) * spacing,
      (column - (side - 1) / 2) * spacing,
    );
  });
}

function layerX(layer: number, layerCount: number) {
  return (layer - (layerCount - 1) / 2) * 8.2;
}

function formatSceneValue(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(5)}`;
}
