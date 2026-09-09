/**
 * Morph Frontend — HTTPS dev server for LAN phone access.
 *
 * Serves Next.js over HTTPS on 0.0.0.0:3000 so that
 * navigator.mediaDevices.getUserMedia() is available on
 * https://192.168.31.103:3000 (secure context).
 *
 * Certs: frontend/certs/morph.key + morph.crt
 *   Generated with: openssl req -x509 -newkey rsa:2048 -days 825 -nodes
 *     -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:192.168.31.103"
 *   The same cert is used for FastAPI WSS (backend/certs/).
 *
 * Usage:
 *   node server.js         # https://0.0.0.0:3000
 *   PORT=3000 node server.js
 */

const { createServer } = require("https");
const { parse } = require("url");
const fs = require("fs");
const path = require("path");
const next = require("next");

const dev = process.env.NODE_ENV !== "production";
const hostname = "0.0.0.0";
const port = parseInt(process.env.PORT || "3000", 10);

const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

const certDir = path.join(__dirname, "certs");
const keyPath = path.join(certDir, "morph.key");
const certPath = path.join(certDir, "morph.crt");

if (!fs.existsSync(keyPath) || !fs.existsSync(certPath)) {
  console.error(`[https] Missing certs: ${keyPath} / ${certPath}`);
  console.error(`[https] Generate with:`);
  console.error(`  openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \\`);
  console.error(`    -keyout frontend/certs/morph.key -out frontend/certs/morph.crt \\`);
  console.error(`    -subj "/CN=192.168.31.103" \\`);
  console.error(`    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:192.168.31.103"`);
  process.exit(1);
}

const httpsOptions = {
  key: fs.readFileSync(keyPath),
  cert: fs.readFileSync(certPath),
};

app.prepare().then(() => {
  createServer(httpsOptions, async (req, res) => {
    try {
      const parsedUrl = parse(req.url, true);
      await handle(req, res, parsedUrl);
    } catch (err) {
      console.error("Error handling", req.url, err);
      res.statusCode = 500;
      res.end("internal server error");
    }
  }).listen(port, hostname, (err) => {
    if (err) throw err;
    console.log(`> Ready on https://${hostname}:${port}`);
    console.log(`> Local:   https://localhost:${port}`);
    // Try to show LAN URL if we can resolve it
    try {
      const os = require("os");
      const ifaces = os.networkInterfaces();
      for (const name of Object.keys(ifaces)) {
        for (const iface of ifaces[name]) {
          if (iface.family === "IPv4" && !iface.internal) {
            console.log(`> Network: https://${iface.address}:${port}`);
          }
        }
      }
    } catch {}
    console.log(`> Phone:   https://192.168.31.103:${port}  (trust cert on phone)`);
  });
});
