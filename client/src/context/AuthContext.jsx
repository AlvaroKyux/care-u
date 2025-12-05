import { createContext, useContext, useEffect, useState } from 'react';
import { setAuthToken } from '../api/http';

const AuthCtx = createContext(null);

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(() => {
    const raw = localStorage.getItem('careu_auth');
    return raw ? JSON.parse(raw) : { user: null, token: null };
  });

  

  const login = (user, token) => {
    localStorage.setItem('careu_auth', JSON.stringify({ user, token }));
    setAuth({ user, token });
    setAuthToken(token);  // Configura el token
  };

  const logout = () => {
    localStorage.removeItem('careu_auth');
    setAuth({ user: null, token: null });
    setAuthToken(null);  // Elimina el token
  };

  return (
    <AuthCtx.Provider value={{ ...auth, login, logout }}>
      {children}
    </AuthCtx.Provider>
  );
}

export const useAuth = () => useContext(AuthCtx);
