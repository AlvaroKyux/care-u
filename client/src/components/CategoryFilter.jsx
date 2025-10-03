export default function CategoryFilter({ categories, value, onChange }){
  return (
    <select className="input" value={value} onChange={e=>onChange(e.target.value)}>
      <option value="all">All categories</option>
      {categories.map(c => <option key={c} value={c}>{c}</option>)}
    </select>
  );
}
