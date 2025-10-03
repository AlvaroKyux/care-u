import { useEffect, useState } from 'react';
import AuthCard from '../components/AuthCard';
import { useAuth } from '../context/AuthContext';
import { getMe, updateMe, changePassword } from '../api/user';

export default function Profile(){
  const { token, login, user } = useAuth();
  const [info, setInfo] = useState({ name:'', email:'' });
  const [pwd, setPwd] = useState({ currentPassword:'', newPassword:'' });
  const [msgI, setMsgI] = useState('');
  const [msgP, setMsgP] = useState('');
  const [loadingI, setLoadingI] = useState(false);
  const [loadingP, setLoadingP] = useState(false);

  useEffect(()=>{
    (async ()=>{
      try{
        const { data } = await getMe(token);
        setInfo({ name:data.user.name, email:data.user.email });
      }catch{}
    })();
  },[token]);

  async function saveProfile(e){
    e.preventDefault(); setMsgI(''); setLoadingI(true);
    try{
      const { data } = await updateMe(info, token);
      // refresca contexto (mantiene token)
      login(data.user, token);
      setMsgI('Perfil actualizado');
    }catch(err){
      const api = err?.response?.data?.error;
      setMsgI(api==='Email already registered' ? 'El correo ya está registrado' : (api || 'Error'));
    }finally{ setLoadingI(false); }
  }

  async function savePassword(e){
    e.preventDefault(); setMsgP(''); setLoadingP(true);
    try{
      await changePassword(pwd, token);
      setPwd({ currentPassword:'', newPassword:'' });
      setMsgP('Contraseña actualizada');
    }catch(err){
      const api = err?.response?.data?.error;
      setMsgP(api==='Wrong current password' ? 'Contraseña actual incorrecta' : (api || 'Error'));
    }finally{ setLoadingP(false); }
  }

  return (
    <div className="container">
      <div className="card" style={{maxWidth:600}}>
        <div className="brand"><span className="dot"/><h1>CARE-U</h1></div>
        <h2>Mi perfil</h2>
        <p className="helper">Edita tu información y contraseña</p>

        <form onSubmit={saveProfile} className="row" style={{gap:12, marginBottom:16}}>
          <input className="input" placeholder="Nombre completo" value={info.name} onChange={e=>setInfo({...info, name:e.target.value})} required />
          <input className="input" type="email" placeholder="Correo" value={info.email} onChange={e=>setInfo({...info, email:e.target.value})} required />
          <button className="btn" disabled={loadingI}>{loadingI?'Guardando…':'Guardar perfil'}</button>
          {msgI && <div className="helper">{msgI}</div>}
        </form>

        <form onSubmit={savePassword} className="row" style={{gap:12}}>
          <input className="input" type="password" placeholder="Contraseña actual" value={pwd.currentPassword} onChange={e=>setPwd({...pwd, currentPassword:e.target.value})} required />
          <input className="input" type="password" placeholder="Nueva contraseña (mín. 6)" value={pwd.newPassword} onChange={e=>setPwd({...pwd, newPassword:e.target.value})} required minLength={6} />
          <button className="btn" disabled={loadingP}>{loadingP?'Actualizando…':'Cambiar contraseña'}</button>
          {msgP && <div className="helper">{msgP}</div>}
        </form>
      </div>
    </div>
  );
}
