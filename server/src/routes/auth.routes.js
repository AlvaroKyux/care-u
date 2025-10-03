import { Router } from 'express';
import { signToken } from '../lib/jwt.js';
import { User, USER_ROLES } from '../models/User.js';

const router = Router();

// HU01 — Registro (Sign up)
router.post('/register', async (req, res) => {
  try {
    const { name, email, password, role } = req.body;

    if (!name || !email || !password || !role) {
      return res.status(400).json({ error: 'Missing fields' });
    }
    if (!USER_ROLES.includes(role)) {
      return res.status(400).json({ error: 'Invalid role' });
    }

    // ¿email ya existe?
    const exists = await User.findOne({ email: email.toLowerCase() }).lean();
    if (exists) return res.status(409).json({ error: 'Email already registered' });

    // Crear usuario (usa virtual "password" para hashear)
    const user = new User({ name, email, role });
    user.password = password;
    await user.save();

    const token = signToken({ id: user._id.toString(), role: user.role });

    // Limpio respuesta
    const { passwordHash, ...safe } = user.toObject();
    return res.status(201).json({ user: safe, token });
  } catch (e) {
    // Detección de índice único duplicate key
    if (e?.code === 11000) {
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

    // Trae también passwordHash con select('+passwordHash')
    const user = await User.findOne({ email: email.toLowerCase() }).select('+passwordHash');
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });

    const ok = await user.comparePassword(password);
    if (!ok) return res.status(401).json({ error: 'Invalid credentials' });

    const token = signToken({ id: user._id.toString(), role: user.role });
    const { passwordHash, ...safe } = user.toObject();
    return res.json({ user: safe, token });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: 'Server error' });
  }
});

export default router;
