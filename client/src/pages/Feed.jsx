import { useEffect, useState } from 'react';
import '../styles/theme.css';
import AuthCard from '../components/AuthCard';
import PostCard from '../components/PostCard';
import CategoryFilter from '../components/CategoryFilter';
import { listCategories, listPosts } from '../api/posts';

export default function Feed(){
  const [cats, setCats] = useState(['maintenance','safety','cleaning','it','other']);
  const [category, setCategory] = useState('all');
  const [q, setQ] = useState('');
  const [items, setItems] = useState([]);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [loading, setLoading] = useState(false);

  useEffect(()=>{
    (async ()=>{
      try{
        const { data } = await listCategories();
        setCats(data.categories);
      }catch{}
    })();
  },[]);

  async function fetchData(p = 1){
    setLoading(true);
    try{
      const { data } = await listPosts({ category, q, page: p, limit: 10 });
      setItems(data.items);
      setPage(data.page);
      setPages(data.pages);
    } finally { setLoading(false); }
  }

  useEffect(()=>{ fetchData(1); }, [category]); // refresca por categoría

  function onSearch(e){
    e.preventDefault();
    fetchData(1);
  }

  return (
    <div className="container">
      <div className="card" style={{maxWidth:860, width:'100%'}}>
        <div className="brand"><span className="dot"/><h1>CARE-U</h1></div>
        <h2>Incidents feed</h2>
        <p className="helper">Filter by category or search text</p>

        <div className="row" style={{gap:12, marginBottom:12}}>
          <CategoryFilter categories={cats} value={category} onChange={setCategory}/>
          <form onSubmit={onSearch} style={{display:'flex', gap:12}}>
            <input className="input" placeholder="Search…" value={q} onChange={e=>setQ(e.target.value)} />
            <button className="btn" disabled={loading}>{loading ? 'Searching…' : 'Search'}</button>
          </form>
        </div>

        <div className="row" style={{gap:14}}>
          {items.map(p => <PostCard key={p._id} post={p} />)}
          {!loading && items.length === 0 && <div className="helper">No results</div>}
        </div>

        <div style={{display:'flex', gap:10, marginTop:16, justifyContent:'flex-end'}}>
          <button className="btn" disabled={page<=1 || loading} onClick={()=>fetchData(page-1)}>Prev</button>
          <button className="btn" disabled={page>=pages || loading} onClick={()=>fetchData(page+1)}>Next</button>
        </div>
      </div>
    </div>
  );
}
