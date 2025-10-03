import '../styles/theme.css';
import { useAuth } from '../context/AuthContext';
import { updatePostStatus } from '../api/posts';
import { useState } from 'react';

const CAT_LABEL = { maintenance:'Mantenimiento', safety:'Seguridad', cleaning:'Limpieza', it:'TI', other:'Otro' };
const STATUS_LABEL = { open:'Abierta', in_progress:'En progreso', resolved:'Resuelta' };

export default function PostCard({ post, onChanged }){
  const { token, user } = useAuth();
  const [loading, setLoading] = useState(false);
  const canManage = user && (user.role === 'admin' || user.role === 'staff');

  async function setStatus(status){
    if(!canManage) return;
    setLoading(true);
    try{
      await updatePostStatus(post._id, status, '', token);
      onChanged?.(); // refrescar lista
    } finally { setLoading(false); }
  }

  return (
    <div style={{ background:'#0c162a', border:'1px solid #1f2a44', borderRadius:12, padding:16, display:'grid', gap:8 }}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:8}}>
        <div style={{fontWeight:700}}>{post.user?.name ?? 'Usuario'}</div>
        <div style={{display:'flex', gap:8, alignItems:'center'}}>
          <span style={{fontSize:12, padding:'4px 8px', borderRadius:999, background:'rgba(97,168,255,.14)', border:'1px solid #284a7a'}}>
            {CAT_LABEL[post.category] || post.category}
          </span>
          <span style={{fontSize:12, padding:'4px 8px', borderRadius:999, background:'rgba(65,209,167,.12)', border:'1px solid #2b6b58'}}>
            {STATUS_LABEL[post.status] || post.status}
          </span>
        </div>
      </div>

      <div style={{whiteSpace:'pre-wrap', lineHeight:1.45}}>{post.text}</div>
      <div className="helper">📍 {post.location?.label} • {new Date(post.createdAt).toLocaleString()}</div>

      {canManage && (
        <div style={{display:'flex', gap:8, justifyContent:'flex-end'}}>
          <button className="btn" disabled={loading || post.status==='in_progress' || post.status==='resolved'} onClick={()=>setStatus('in_progress')}>Marcar en progreso</button>
          <button className="btn" disabled={loading || post.status==='resolved'} onClick={()=>setStatus('resolved')}>Marcar resuelta</button>
        </div>
      )}
    </div>
  );
}
