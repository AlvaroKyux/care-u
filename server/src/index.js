import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import cookieParser from 'cookie-parser';

const app = express();

// Middlewares
app.use(cors({ origin: 'http://localhost:5173', credentials: true }));
app.use(express.json());
app.use(cookieParser());

// Healthcheck
app.get('/api/health', (req, res) => {
  res.json({ ok: true, service: 'CARE-U API', timestamp: new Date().toISOString() });
});

// (Más adelante agregaremos /api/auth para HU01/HU02)
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`✅ CARE-U API escuchando en http://localhost:${PORT}`);
});
