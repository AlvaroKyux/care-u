import { Router } from 'express';
import { requireAuth } from '../middlewares/requireAuth.js';
import { Post, CATEGORIES } from '../models/Post.js';

const router = Router();

/** HU03 — Crear incidencia */
router.post('/', requireAuth, async (req, res) => {
  try {
    const { text, category, location } = req.body;
    if (!text || !category || !location?.label) {
      return res.status(400).json({ error: 'Missing fields' });
    }
    if (!CATEGORIES.includes(category)) {
      return res.status(400).json({ error: 'Invalid category' });
    }

    const doc = await Post.create({
      user: req.user.id,
      text, category,
      location: {
        label: String(location.label),
        lat: location.lat ?? undefined,
        lng: location.lng ?? undefined
      }
    });

    const populated = await doc.populate('user', 'name role');
    res.status(201).json({ post: populated });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Server error' });
  }
});

/** HU04 — Listar feed con filtros y paginación */
router.get('/', async (req, res) => {
  try {
    const { category, q, page = 1, limit = 10 } = req.query;
    const filt = {};
    if (category && category !== 'all') filt.category = category;
    if (q) filt.$text = { $search: q };

    const skip = (Number(page) - 1) * Number(limit);

    const [items, total] = await Promise.all([
      Post.find(filt)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(Number(limit))
        .populate('user', 'name role')
        .lean(),
      Post.countDocuments(filt)
    ]);

    res.json({
      items,
      total,
      page: Number(page),
      pages: Math.ceil(total / Number(limit))
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Server error' });
  }
});

/** Listado de categorías (para el selector del frontend) */
router.get('/categories', (_req, res) => {
  res.json({ categories: CATEGORIES });
});

export default router;
