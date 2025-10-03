import { Router } from 'express';
import { createUser, verifyCredentials } from '../models/userStore.js';
import { signToken } from '../lib/jwt.js';

const router = Router();

// HU01 — Registro
router.post('/register', async (req, res) => {
  try {
    const { name, email, password, role } = req.body;

    if (!name || !email || !password || !role) {
      return res.status(400).json({ error: 'Missing fields' });
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      return res.status(400).json({ error: 'Invalid email' });
    }
    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be 6+ chars' });
    }
    if (!['admin', 'staff', 'student'].includes(role)) {
      return res.status(400).json({ error: 'Invalid role' });
    }

    const user = await createUser({ name, email, password, role });
    const token = signToken({ id: user.id, role: user.role });

    // Simple: devolver token en JSON (rápido para pruebas)
    return res.status(201).json({ user, token });
  } catch (e) {
    if (e.code === 'EMAIL_TAKEN') {
      return res.status(409).json({ error: 'Email already registered' });
    }
    console.error(e);
    return res.status(500).json({ error: 'Server error' });
  }
});

// HU02 — Login
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ error: 'Missing credentials' });

    const user = await verifyCredentials(email, password);
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });

    const token = signToken({ id: user.id, role: user.role });
    return res.json({ user, token });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: 'Server error' });
  }
});

export default router;
