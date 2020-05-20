const tf = require('@tensorflow/tfjs-node');
const bodyPix = require('@tensorflow-models/body-pix');
const http = require('http');
const util = require('util');
const zlib = require('zlib');
const gzip = util.promisify(zlib.gzip);

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
                internalResolution: 'full',
                segmentationThreshold: 0.7,
            });
            zipped = await gzip(Buffer.from(segmentation.data));
            res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Encoding': 'gzip' });
            res.write(zipped);
            res.end();
            tf.dispose(image);
        });
    });
    server.listen(9000);
})();
