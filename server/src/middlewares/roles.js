export function requireRole(...roles){
  return (req, res, next)=>{
    // viene de requireAuth (ya lo tienes)
    if(!req.user) return res.status(401).json({ error:'No token' });
    if(!roles.includes(req.user.role)) return res.status(403).json({ error:'Forbidden' });
    next();
  };
}
