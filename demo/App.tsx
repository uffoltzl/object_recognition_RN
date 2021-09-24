import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, Platform } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from '@tensorflow-models/mobilenet';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';

const textureDims = Platform.OS === 'ios' ?
  {
    height: 1920,
    width: 1080,
  } :
   {
    height: 1200,
    width: 1600,
  };

let frame = 0;
const computeRecognitionEveryNFrames = 60;

const TensorCamera = cameraWithTensors(Camera);

const loadModel = async () => {
  await tf.ready();
  tf.getBackend();
}

export default function App() {
  const [hasPermission, setHasPermission] = useState<null | boolean>(null);
  const [detections, setDetections] = useState<string[]>([]);
  const [net, setNet] = useState<mobilenet.MobileNet>();


  const handleCameraStream = (images: IterableIterator<tf.Tensor3D>) => {
    const loop = async () => {
      if(net) {
        if(frame % computeRecognitionEveryNFrames === 0){
          const nextImageTensor = images.next().value;
          if(nextImageTensor){
            const objects = await net.classify(nextImageTensor);
            if(objects && objects.length > 0){
              setDetections(objects.map(object => object.className));
            }
            tf.dispose([nextImageTensor]);
          }
        }
        frame += 1;
        frame = frame % computeRecognitionEveryNFrames;
      }

      requestAnimationFrame(loop);
    }
    loop();
  }

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestPermissionsAsync();
      setHasPermission(status === 'granted');
      await loadModel();
      setNet(await mobilenet.load({version: 1, alpha: 0.25}));
    })();
  }, []);

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }
  if(!net){
    return <Text>Model not loaded</Text>;
  }

  return (
    <View style={styles.container}>
      <TensorCamera 
        style={styles.camera} 
        onReady={handleCameraStream}
        type={Camera.Constants.Type.back}
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={200}
        resizeWidth={152}
        resizeDepth={3}
        autorender={true}
      />
      <View style={styles.text}>
      {detections.map((detection, index) => 
          <Text key={index}>{detection}</Text>
      )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    flex: 1,
  },
  camera: {
    flex: 10,
    width: '100%',
  },
});
