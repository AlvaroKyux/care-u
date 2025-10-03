import { useEffect, useRef, useState } from 'react';
import '../styles/theme.css';
import PostCard from '../components/PostCard';
import CategoryFilter from '../components/CategoryFilter';
import { listCategories, listPosts } from '../api/posts';
import NotificationsBell from '../components/NotificationsBell';

export default function Feed(){
  const [cats, setCats] = useState(['maintenance','safety','cleaning','it','other']);
  const [category, setCategory] = useState('all');
  const [q, setQ] = useState('');
  const [items, setItems] = useState([]);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const mounted = useRef(false);

  useEffect(()=>{
    (async ()=>{
      try{
        const { data } = await listCategories();
        if (Array.isArray(data?.categories)) setCats(data.categories);
      }catch{}
    })();
  },[]);

  async function fetchData(p = 1){
    setLoading(true);
    setErr('');
    try{
      const { data } = await listPosts({ category, q, page: p, limit: 10 });
      setItems(data.items || []);
      setPage(Number(data.page) || p);
      setPages(Number(data.pages) || 1);
    } catch {
      setErr('Error al cargar el feed');
      setItems([]);
      setPage(1);
      setPages(1);
    } finally {
      setLoading(false);
    }
  }

  useEffect(()=>{
    if (!mounted.current) {
      mounted.current = true;
      fetchData(1);
      return;
    }
    fetchData(1);
  }, [category]);

  function onSearch(e){
    e.preventDefault();
    fetchData(1);
  }

  return (
    <div className="container">
      <div className="card" style={{maxWidth:860, width:'100%'}}>
        <div className="brand"><span className="dot"/><h1>CARE-U</h1></div>
        <h2>Feed de incidencias</h2>
        <p className="helper">Filtra por categoría o busca por texto</p>

        <div className="row" style={{gap:12, marginBottom:12}}>
          <CategoryFilter categories={cats} value={category} onChange={setCategory}/>
          <form onSubmit={onSearch} style={{display:'flex', gap:12}}>
            <input
              className="input"
              placeholder="Buscar…"
              value={q}
              onChange={e=>setQ(e.target.value)}
            />
            <button className="btn" disabled={loading}>
              {loading ? 'Buscando…' : 'Buscar'}
            </button>
          </form>
        </div>

        {err && <div className="error" style={{marginBottom:8}}>{err}</div>}

        <div className="row" style={{gap:14}}>
          {items.map(p => (
            <PostCard
              key={p._id}
              post={p}
              onChanged={() => fetchData(page)}
            />
          ))}
          {!loading && items.length === 0 && <div className="helper">Sin resultados</div>}
        </div>

        <div style={{display:'flex', gap:10, marginTop:16, justifyContent:'flex-end'}}>
          <button className="btn" disabled={page<=1 || loading} onClick={()=>fetchData(page-1)}>Anterior</button>
          <button className="btn" disabled={page>=pages || loading} onClick={()=>fetchData(page+1)}>Siguiente</button>
        </div>
      </div>

      {/* Campana flotante */}
      <NotificationsBell />
    </div>
  );
}
