import { useEffect, useState } from 'react';
import AuthCard from '../components/AuthCard';
import { createPost } from '../api/posts';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

const CATEGORIES = ['maintenance','safety','cleaning','it','other'];

export default function CreateIncident(){
  const { token } = useAuth();
  const nav = useNavigate();
  const [form, setForm] = useState({
    text:'', category:'maintenance',
    locationLabel:'', lat:'', lng:''
  });
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState({ type:null, text:'' });

  // autollenar lat/lng (opcional)
  useEffect(()=>{
    if(!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos)=>{
        setForm(f=>({
          ...f,
          lat: pos.coords.latitude.toFixed(6),
          lng: pos.coords.longitude.toFixed(6)
        }));
      },
      ()=>{}
    );
  },[]);

  const onChange = e => setForm(f=>({ ...f, [e.target.name]: e.target.value }));

  async function onSubmit(e){
    e.preventDefault(); setMsg({type:null,text:''}); setLoading(true);
    try{
      const payload = {
        text: form.text.trim(),
        category: form.category,
        location: {
          label: form.locationLabel.trim(),
          lat: form.lat ? Number(form.lat) : undefined,
          lng: form.lng ? Number(form.lng) : undefined
        }
      };
      await createPost(payload, token);
      nav('/feed');
    }catch(err){
      const text = err?.response?.data?.error || 'Error';
      setMsg({type:'error', text});
    }finally{ setLoading(false); }
  }

  return (
    <AuthCard title="Report an incident" subtitle="Describe, categorize and locate the problem">
      <form onSubmit={onSubmit} className="row" style={{gap:14}}>
        <textarea
          className="input" name="text" rows={4} placeholder="What happened?"
          value={form.text} onChange={onChange} required minLength={5} maxLength={500}
          style={{resize:'vertical'}}
        />
        <div className="row cols-2">
          <select className="input" name="category" value={form.category} onChange={onChange}>
            {CATEGORIES.map(c=><option key={c} value={c}>{c}</option>)}
          </select>
          <input className="input" name="locationLabel" placeholder="Location (e.g., Building A, Lab 3)"
                 value={form.locationLabel} onChange={onChange} required />
        </div>
        <div className="row cols-2">
          <input className="input" name="lat" placeholder="Latitude (optional)" value={form.lat} onChange={onChange} />
          <input className="input" name="lng" placeholder="Longitude (optional)" value={form.lng} onChange={onChange} />
        </div>
        <button className="btn" disabled={loading}>{loading ? 'Posting…':'Create incident'}</button>
        {msg.text && <div className={msg.type==='error'?'error':'success'}>{msg.text}</div>}
      </form>
    </AuthCard>
  );
}
