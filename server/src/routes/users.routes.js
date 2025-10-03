import { Router } from 'express';
import { requireAuth } from '../middlewares/requireAuth.js';
import { User } from '../models/User.js';
import bcrypt from 'bcryptjs';

const router = Router();

// Obtener mi perfil
router.get('/me', requireAuth, async (req,res)=>{
  const user = await User.findById(req.user.id).lean();
  if(!user) return res.status(404).json({ error:'Not found' });
  res.json({ user });
});

// Actualizar nombre/correo
router.put('/me', requireAuth, async (req,res)=>{
  try{
    const { name, email } = req.body;
    if(!name || !email) return res.status(400).json({ error:'Missing fields' });

    const exists = await User.findOne({ _id: { $ne: req.user.id }, email: email.toLowerCase() }).lean();
    if(exists) return res.status(409).json({ error: 'Email already registered' });

    const updated = await User.findByIdAndUpdate(
      req.user.id,
      { $set: { name, email: email.toLowerCase() } },
      { new:true }
    ).lean();

    res.json({ user: updated });
  }catch(e){
    console.error(e);
    res.status(500).json({ error:'Server error' });
  }
});

// Cambiar contraseña (requiere contraseña actual)
router.put('/me/password', requireAuth, async (req,res)=>{
  const { currentPassword, newPassword } = req.body;
  if(!currentPassword || !newPassword) return res.status(400).json({ error:'Missing fields' });
  if(newPassword.length < 6) return res.status(400).json({ error:'Password must be 6+ chars' });

  const user = await User.findById(req.user.id).select('+passwordHash');
  if(!user) return res.status(404).json({ error: 'Not found' });

  const ok = await user.comparePassword(currentPassword);
  if(!ok) return res.status(401).json({ error: 'Wrong current password' });

  user.password = newPassword;         // virtual → se hashea
  await user.save();

  res.json({ ok:true });
});

export default router;
