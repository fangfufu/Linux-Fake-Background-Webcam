
const PORT = process.env.PORT || 9000;
const tf = tensorflow();

const bodyPix = require('@tensorflow-models/body-pix');
const http = require('http');
(async () => {
    const net = await bodyPix.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        multiplier: 0.75,
        quantBytes: 2,
    });
    const server = http.createServer();
    server.on('request', async (req, res) => {
        var chunks = [];
        req.on('data', (chunk) => {
            chunks.push(chunk);
        });
        req.on('end', async () => {
            const image = tf.node.decodeImage(Buffer.concat(chunks));
            segmentation = await net.segmentPerson(image, {
                flipHorizontal: false,
                internalResolution: 'medium',
                segmentationThreshold: 0.75,
            });
            res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
            res.write(Buffer.from(segmentation.data));
            res.end();
            tf.dispose(image);
        });
    });
    server.listen(PORT);
})();


function tensorflow() {
    const GPU = process.env.GPU || "/dev/nvidia0";
    const fs = require('fs')
    if (fs.existsSync(GPU)) {
        console.log('Found a GPU at %s', GPU);
        return require('@tensorflow/tfjs-node-gpu');
    } else {
        console.log('No GPU found at %s, using CPU', GPU);
        return require('@tensorflow/tfjs-node');
    }
}