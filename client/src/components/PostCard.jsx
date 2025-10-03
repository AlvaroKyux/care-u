import '../styles/theme.css';

export default function PostCard({ post }){
  return (
    <div style={{
      background:'#0c162a', border:'1px solid #1f2a44',
      borderRadius:12, padding:16, display:'grid', gap:8
    }}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <div style={{fontWeight:700}}>{post.user?.name ?? 'User'}</div>
        <span style={{
          fontSize:12, padding:'4px 8px', borderRadius:999,
          background:'rgba(97,168,255,.14)', border:'1px solid #284a7a'
        }}>{post.category}</span>
      </div>
      <div style={{whiteSpace:'pre-wrap', lineHeight:1.45}}>{post.text}</div>
      <div className="helper">
        📍 {post.location?.label} • {new Date(post.createdAt).toLocaleString()}
      </div>
    </div>
  );
}
