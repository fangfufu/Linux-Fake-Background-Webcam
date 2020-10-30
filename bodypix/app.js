
const HFLIP = process.env.BPHFLIP || false;
const IRES = parseFloat(process.env.BPIRES) || 0.5;
const MULTI = parseFloat(process.env.BPMULTI) || 0.75;
const OUTSTRIDE = parseInt(process.env.BPOUTSTRIDE) || 16;
const PORT = process.env.BPPORT || 9000;
const QBYTES = parseInt(process.env.BPQBYTES) || 2;
const SEGTHRES = parseFloat(process.env.BPSEGTHRES) || 0.75;
const tf = tensorflow();

const bodyPix = require('@tensorflow-models/body-pix');
const http = require('http');
(async () => {
    const net = await bodyPix.load({
        architecture: 'MobileNetV1',
        outputStride: OUTSTRIDE,
        multiplier: MULTI,
        quantBytes: QBYTES,
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
                flipHorizontal: HFLIP,
                internalResolution: IRES,
                segmentationThreshold: SEGTHRES,
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
