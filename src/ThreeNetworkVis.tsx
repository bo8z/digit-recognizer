import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import GUI from "lil-gui";
import type { ForwardResult, NetworkData } from "./nn";

interface SceneMeshes {
  inputMesh: THREE.InstancedMesh;
  outputMesh: THREE.InstancedMesh;
  connectionOpacity: THREE.BufferAttribute;
  opacityReference: number;
}

export function ThreeNetworkVis({
  result,
  networkData,
}: {
  result: ForwardResult | null;
  networkData: NetworkData;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const meshesRef = useRef<SceneMeshes | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const inputCount = networkData.sizes[0];
    const outputCount = networkData.sizes[networkData.sizes.length - 1];
    const gridSize = Math.sqrt(inputCount);
    const inputSpacing = 0.5;
    const outputSpacing = 1.35;
    const outputZ = 12;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);

    const initialCameraPosition = new THREE.Vector3(15, 10, 23);
    const initialTarget = new THREE.Vector3(0, 0, outputZ / 2);
    const orthographicViewHeight = 20;
    const perspectiveCamera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    const orthographicCamera = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 100);
    perspectiveCamera.position.copy(initialCameraPosition);
    orthographicCamera.position.copy(initialCameraPosition);
    let activeCamera: THREE.PerspectiveCamera | THREE.OrthographicCamera = orthographicCamera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
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
    orthographicCamera.zoom = orthographicViewHeight
      / (2 * initialDistance * Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov / 2)));
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
        orthographicCamera.zoom = orthographicViewHeight
          / (2 * initialDistance * Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov / 2)));
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
        const direction = activeCamera.position.clone().sub(controls.target).normalize();
        activeCamera.position.copy(controls.target).addScaledVector(direction, distance);
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
          orthographicCamera.zoom = orthographicViewHeight
            / (2 * distance * Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov / 2)));
          cameraSettings.orthographicZoom = orthographicCamera.zoom;
        } else {
          const distance = orthographicViewHeight
            / (2 * orthographicCamera.zoom * Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov / 2)));
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

    scene.add(new THREE.AmbientLight(0xffffff, 1.8));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 2.8);
    directionalLight.position.set(8, 12, 18);
    scene.add(directionalLight);

    const inputGeometry = new THREE.BoxGeometry(0.36, 0.36, 0.24);
    const inputMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      metalness: 0.15,
      roughness: 0.65,
    });
    const inputMesh = new THREE.InstancedMesh(inputGeometry, inputMaterial, inputCount);
    const inputPositions: THREE.Vector3[] = new Array(inputCount);
    const transform = new THREE.Object3D();
    const inactiveInputColor = new THREE.Color(0x20262e);

    for (let index = 0; index < inputCount; index++) {
      const row = Math.floor(index / gridSize);
      const column = index % gridSize;
      const position = new THREE.Vector3(
        (column - (gridSize - 1) / 2) * inputSpacing,
        ((gridSize - 1) / 2 - row) * inputSpacing,
        0,
      );
      inputPositions[index] = position;
      transform.position.copy(position);
      transform.updateMatrix();
      inputMesh.setMatrixAt(index, transform.matrix);
      inputMesh.setColorAt(index, inactiveInputColor);
    }
    inputMesh.instanceMatrix.needsUpdate = true;
    if (inputMesh.instanceColor) inputMesh.instanceColor.needsUpdate = true;
    scene.add(inputMesh);

    const outputGeometry = new THREE.BoxGeometry(0.9, 0.9, 0.9);
    const outputMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      metalness: 0.25,
      roughness: 0.4,
    });
    const outputMesh = new THREE.InstancedMesh(outputGeometry, outputMaterial, outputCount);
    const outputPositions: THREE.Vector3[] = new Array(outputCount);
    const inactiveOutputColor = new THREE.Color(0x28313b);
    const labelSprites: THREE.Sprite[] = [];

    for (let index = 0; index < outputCount; index++) {
      const position = new THREE.Vector3(
        0,
        ((outputCount - 1) / 2 - index) * outputSpacing,
        outputZ,
      );
      outputPositions[index] = position;
      transform.position.copy(position);
      transform.updateMatrix();
      outputMesh.setMatrixAt(index, transform.matrix);
      outputMesh.setColorAt(index, inactiveOutputColor);

      const label = createDigitLabel(index);
      label.position.set(0.85, position.y, outputZ);
      labelSprites.push(label);
      scene.add(label);
    }
    outputMesh.instanceMatrix.needsUpdate = true;
    if (outputMesh.instanceColor) outputMesh.instanceColor.needsUpdate = true;
    scene.add(outputMesh);

    const weightMagnitudes = networkData.weights[0]
      .flat()
      .map((weight) => Math.abs(weight))
      .sort((a, b) => a - b);
    const opacityReference = weightMagnitudes[Math.floor((weightMagnitudes.length - 1) * 0.99)];

    const connectionCount = inputCount * outputCount;
    const positions = new Float32Array(connectionCount * 6);
    const colors = new Float32Array(connectionCount * 6);
    const opacities = new Float32Array(connectionCount * 2).fill(0.008);
    const positiveColor = new THREE.Color(0x58a6ff);
    const negativeColor = new THREE.Color(0xff6b63);
    let offset = 0;

    for (let outputIndex = 0; outputIndex < outputCount; outputIndex++) {
      for (let inputIndex = 0; inputIndex < inputCount; inputIndex++) {
        const inputPosition = inputPositions[inputIndex];
        const outputPosition = outputPositions[outputIndex];
        const weight = networkData.weights[0][outputIndex][inputIndex];
        const color = weight >= 0 ? positiveColor : negativeColor;

        positions.set(
          [
            inputPosition.x,
            inputPosition.y,
            inputPosition.z,
            outputPosition.x,
            outputPosition.y,
            outputPosition.z,
          ],
          offset,
        );
        colors.set([color.r, color.g, color.b, color.r, color.g, color.b], offset);
        offset += 6;
      }
    }

    const connectionGeometry = new THREE.BufferGeometry();
    connectionGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    connectionGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const connectionOpacity = new THREE.BufferAttribute(opacities, 1);
    connectionOpacity.setUsage(THREE.DynamicDrawUsage);
    connectionGeometry.setAttribute("connectionOpacity", connectionOpacity);
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
    scene.add(new THREE.LineSegments(connectionGeometry, connectionMaterial));

    meshesRef.current = { inputMesh, outputMesh, connectionOpacity, opacityReference };

    const resizeObserver = new ResizeObserver(() => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      const aspect = width / height;
      perspectiveCamera.aspect = aspect;
      perspectiveCamera.updateProjectionMatrix();
      orthographicCamera.left = -orthographicViewHeight * aspect / 2;
      orthographicCamera.right = orthographicViewHeight * aspect / 2;
      orthographicCamera.top = orthographicViewHeight / 2;
      orthographicCamera.bottom = -orthographicViewHeight / 2;
      orthographicCamera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    });
    resizeObserver.observe(mount);

    let animationFrame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, activeCamera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      gui.destroy();
      inputGeometry.dispose();
      inputMaterial.dispose();
      outputGeometry.dispose();
      outputMaterial.dispose();
      connectionGeometry.dispose();
      connectionMaterial.dispose();
      for (const label of labelSprites) {
        label.material.map?.dispose();
        label.material.dispose();
      }
      renderer.dispose();
      renderer.domElement.remove();
      meshesRef.current = null;
    };
  }, [networkData]);

  useEffect(() => {
    const meshes = meshesRef.current;
    if (!meshes) return;

    const inputActivations = result?.activations[0];
    const outputActivations = result?.activations[result.activations.length - 1];
    const inputColor = new THREE.Color();
    const outputColor = new THREE.Color();
    const inactiveInputColor = new THREE.Color(0x20262e);
    const activeInputColor = new THREE.Color(0xffffff);
    const inactiveOutputColor = new THREE.Color(0x28313b);
    const activeOutputColor = new THREE.Color(0x58a6ff);
    const connectionOpacities = meshes.connectionOpacity.array as Float32Array;
    const inputCount = networkData.sizes[0];
    const outputCount = networkData.sizes[networkData.sizes.length - 1];

    for (let index = 0; index < meshes.inputMesh.count; index++) {
      const activation = inputActivations?.[index] ?? 0;
      inputColor.copy(inactiveInputColor).lerp(activeInputColor, activation);
      meshes.inputMesh.setColorAt(index, inputColor);
    }
    if (meshes.inputMesh.instanceColor) meshes.inputMesh.instanceColor.needsUpdate = true;

    for (let index = 0; index < meshes.outputMesh.count; index++) {
      const activation = outputActivations?.[index] ?? 0;
      outputColor.copy(inactiveOutputColor).lerp(activeOutputColor, activation);
      meshes.outputMesh.setColorAt(index, outputColor);
    }
    if (meshes.outputMesh.instanceColor) meshes.outputMesh.instanceColor.needsUpdate = true;

    for (let outputIndex = 0; outputIndex < outputCount; outputIndex++) {
      for (let inputIndex = 0; inputIndex < inputCount; inputIndex++) {
        const inputActivation = inputActivations?.[inputIndex] ?? 0;
        const weight = networkData.weights[0][outputIndex][inputIndex];
        const contribution = Math.abs(inputActivation * weight);
        const normalizedContribution = meshes.opacityReference > 0
          ? Math.min(contribution / meshes.opacityReference, 1)
          : 0;
        const opacity = 0.008 + Math.pow(normalizedContribution, 0.8) * 0.72;
        const connectionIndex = outputIndex * inputCount + inputIndex;
        connectionOpacities[connectionIndex * 2] = opacity;
        connectionOpacities[connectionIndex * 2 + 1] = opacity;
      }
    }
    meshes.connectionOpacity.needsUpdate = true;
  }, [result, networkData]);

  return (
    <section
      style={{
        flex: "1 1 800px",
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "8px",
      }}
    >
      <h2 style={{ fontSize: "18px", fontWeight: 500, margin: 0 }}>
        Full 3D network
      </h2>
      <p style={{ color: "#8b949e", fontSize: "13px", margin: 0 }}>
        784 input blocks · 7,840 weighted connections · 10 output blocks
      </p>
      <p style={{ color: "#8b949e", fontSize: "12px", margin: 0 }}>
        Blue = positive contribution · Red = negative contribution · Opacity = live signal magnitude
      </p>
      <p style={{ color: "#484f58", fontSize: "12px", margin: "0 0 8px" }}>
        Drag to orbit · Scroll to zoom · Right-drag to pan
      </p>
      <div
        ref={mountRef}
        role="region"
        aria-label="Interactive three-dimensional view of all 784 input nodes, 10 output nodes, and 7,840 connections"
        style={{
          width: "100%",
          height: "clamp(440px, 58vw, 680px)",
          position: "relative",
          border: "1px solid #30363d",
          borderRadius: "8px",
          overflow: "hidden",
        }}
      />
    </section>
  );
}

function createDigitLabel(digit: number): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext("2d")!;
  context.fillStyle = "#e6edf3";
  context.font = "500 36px system-ui";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(String(digit), 32, 32);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.8, 0.8, 1);
  return sprite;
}
