import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const CLASES = ['negativo', 'neutro', 'positivo']
const EJEMPLOS = [
  'Habitación sucia y personal grosero, no volvería nunca',
  'El hotel está bien, aunque el desayuno era normalito',
  'Excelente ubicación, trato amable y camas comodísimas',
]
// Secciones de la página (una sola página con scroll, como la plantilla iPortfolio)
const SECCIONES = [
  { id: 'inicio', titulo: 'Inicio', icono: 'casa' },
  { id: 'analizar', titulo: 'Analizar reseña', icono: 'mensaje' },
  { id: 'metricas', titulo: 'Métricas', icono: 'grafico' },
  { id: 'funcionamiento', titulo: 'Cómo funciona', icono: 'info' },
]
const ENLACES = [
  { titulo: 'Repositorio en GitHub', href: 'https://github.com/Elizadev21/REPO-PRUEBA-TOPICOS', icono: 'github' },
  { titulo: 'Documentación de la API (Swagger)', href: '/docs', icono: 'codigo' },
  { titulo: 'Documentación de la API (ReDoc)', href: '/redoc', icono: 'libro' },
]
const PALABRAS_PORTADA = ['negativas', 'neutras', 'positivas']
// Debe coincidir con el punto de corte xl: de Tailwind (1280 px), donde el menú queda fijo
const ESCRITORIO = '(min-width: 1280px)'
const esEscritorio = () => window.matchMedia(ESCRITORIO).matches
// Se redondea sobre el valor en milésimas para evitar errores de coma flotante (0.7855 -> 78.6 %)
const porcentaje = (valor) => `${(Math.round(Number(valor) * 1000) / 10).toFixed(1)} %`

// Clases completas (no construidas por partes) para que Tailwind las detecte al compilar
const FONDO_CLASE = { negativo: 'bg-negativo', neutro: 'bg-neutro', positivo: 'bg-positivo' }
const TARJETA = 'rounded-lg bg-white p-6 shadow-[0_0_24px_rgba(4,11,20,0.08)]'
const NOTA = 'text-sm text-slate-500'

async function pedir(ruta, opciones) {
  let respuesta
  try { respuesta = await fetch(`/api${ruta}`, opciones) }
  catch { throw new Error('No se pudo conectar con la API.') }
  const datos = await respuesta.json().catch(() => ({}))
  if (respuesta.status === 422) throw new Error(typeof datos.detail === 'string' ? datos.detail : 'Revisa la reseña e inténtalo de nuevo.')
  if (!respuesta.ok) throw new Error(typeof datos.detail === 'string' ? datos.detail : 'La API no está disponible.')
  return datos
}

function Icono({ nombre, className = 'size-5' }) {
  const trazos = {
    menu: <path d="M4 6h16M4 12h16M4 18h16" />,
    cerrar: <path d="M6 6l12 12M18 6 6 18" />,
    casa: <><path d="M4 11 12 4l8 7" /><path d="M6 10v10h12V10" /></>,
    mensaje: <><path d="M4 5h16v11H9l-5 4z" /><path d="M8 9h8M8 12h5" /></>,
    grafico: <path d="M4 20h16M7 16v-5M12 16V6M17 16v-8" />,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></>,
    codigo: <path d="m8 8-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14" />,
    libro: <><path d="M5 4h10a4 4 0 0 1 4 4v12H9a4 4 0 0 1-4-4z" /><path d="M5 16a4 4 0 0 1 4-4h10" /></>,
    github: <path d="M9 19c-4 1.3-4-2-6-2.5m12 5v-3.4a3 3 0 0 0-.8-2.3c2.7-.3 5.6-1.3 5.6-6a4.7 4.7 0 0 0-1.3-3.2 4.3 4.3 0 0 0-.1-3.2s-1-.3-3.4 1.3a11.6 11.6 0 0 0-6 0C6.6 2.9 5.6 3.2 5.6 3.2a4.3 4.3 0 0 0-.1 3.2A4.7 4.7 0 0 0 4.2 9.6c0 4.6 2.9 5.7 5.6 6a3 3 0 0 0-.8 2.3V22" />,
    flecha: <path d="M12 5v14m-6-6 6 6 6-6" />,
  }
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{trazos[nombre]}</svg>
  )
}

/** Escribe y borra las palabras una a una, como el "I'm Developer|" de la plantilla. */
function useEscritura(palabras) {
  const [texto, setTexto] = useState('')
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setTexto(palabras[0]); return }
    let indice = 0, letras = 0, borrando = false, temporizador
    const paso = () => {
      const palabra = palabras[indice]
      letras += borrando ? -1 : 1
      setTexto(palabra.slice(0, letras))
      let espera = borrando ? 50 : 100
      if (!borrando && letras === palabra.length) { borrando = true; espera = 1800 }
      else if (borrando && letras === 0) { borrando = false; indice = (indice + 1) % palabras.length; espera = 400 }
      temporizador = setTimeout(paso, espera)
    }
    temporizador = setTimeout(paso, 600)
    return () => clearTimeout(temporizador)
  }, [palabras])
  return texto
}

/** Devuelve el id de la sección que ocupa el centro de la pantalla (para resaltar el menú). */
function useSeccionVisible(ids) {
  const [activa, setActiva] = useState(ids[0])
  useEffect(() => {
    const observador = new IntersectionObserver(
      (entradas) => entradas.forEach((e) => e.isIntersecting && setActiva(e.target.id)),
      { rootMargin: '-50% 0px -50% 0px' },
    )
    ids.forEach((id) => { const el = document.getElementById(id); if (el) observador.observe(el) })
    return () => observador.disconnect()
  }, [ids])
  return activa
}

function TituloSeccion({ children, descripcion }) {
  return (
    <div className="mb-8">
      <h2 className="relative pb-4 font-titulo text-3xl font-bold text-titulo after:absolute after:bottom-0 after:left-0 after:h-[3px] after:w-12 after:bg-acento">
        {children}
      </h2>
      {descripcion && <p className="mt-4 max-w-3xl text-slate-600">{descripcion}</p>}
    </div>
  )
}

function Seccion({ id, alterna, children }) {
  return (
    <section id={id} className={`scroll-mt-0 px-6 py-16 sm:px-10 lg:px-16 ${alterna ? 'bg-fondo-alt' : 'bg-white'}`}>
      <div className="mx-auto max-w-5xl">{children}</div>
    </section>
  )
}

function Barras({ probabilidades }) {
  return (
    <div className="space-y-4">
      {CLASES.map((c) => (
        <div key={c}>
          <div className="mb-1 flex justify-between font-menu text-sm font-medium uppercase tracking-wide text-titulo">
            <span>{c}</span><span className="tabular-nums">{porcentaje(probabilidades[c])}</span>
          </div>
          <div className="h-2.5 overflow-hidden bg-slate-200">
            <div className={`h-full transition-[width] duration-700 ${FONDO_CLASE[c]}`} style={{ width: `${probabilidades[c] * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}

function Fichas({ items, tipo, vacio }) {
  if (!items.length) return <span className={NOTA}>{vacio}</span>
  const estilo = {
    quitada: 'border-slate-200 bg-slate-50 text-slate-400 line-through',
    raiz: 'border-acento/30 bg-acento/10 text-acento-oscuro',
  }[tipo] || 'border-slate-200 bg-slate-100 text-slate-700'
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((t, i) => <span key={`${t}-${i}`} className={`rounded border px-2 py-0.5 font-mono text-sm ${estilo}`}>{t}</span>)}
    </div>
  )
}

function Preprocesamiento({ datos }) {
  const etapas = [
    ['Texto normalizado', 'Minúsculas, sin tildes, números, signos ni URLs.',
      <code key="n" className="font-mono text-sm break-words text-slate-700">{datos.normalizado || '—'}</code>],
    ['Tokens', 'Palabras separadas (se descartan las de una letra).', <Fichas key="t" items={datos.tokens} vacio="Sin tokens" />],
    ['Stop words eliminadas', 'Palabras funcionales; se conservan negaciones como «no».',
      <Fichas key="s" items={datos.stopwords_eliminadas} tipo="quitada" vacio="Ninguna" />],
    ['Raíces (stemming)', 'Lo que recibe el TF-IDF del modelo.', <Fichas key="r" items={datos.raices} tipo="raiz" vacio="Sin raíces" />],
  ]
  return (
    <div className={`${TARJETA} mt-6`}>
      <h3 className="mb-5 font-titulo text-xl font-bold text-titulo">Preprocesamiento paso a paso</h3>
      {/* Línea de tiempo vertical, como la sección "Resume" de la plantilla */}
      <ol className="border-l-2 border-acento">
        {etapas.map(([titulo, ayuda, contenido]) => (
          <li key={titulo} className="relative pb-6 pl-6 last:pb-0 before:absolute before:top-1 before:-left-[9px] before:size-4 before:rounded-full before:border-2 before:border-acento before:bg-white">
            <p className="font-titulo font-bold text-titulo uppercase">{titulo}</p>
            <p className={`${NOTA} mb-2 italic`}>{ayuda}</p>
            {contenido}
          </li>
        ))}
      </ol>
    </div>
  )
}

function Analizar() {
  const [texto, setTexto] = useState('')
  const [resultado, setResultado] = useState(null)
  const [error, setError] = useState('')
  const [ocupado, setOcupado] = useState(false)

  async function analizar(evento) {
    evento.preventDefault()
    if (!texto.trim()) { setError('Escribe una reseña antes de analizar.'); return }
    setOcupado(true); setError(''); setResultado(null)
    try {
      setResultado(await pedir('/predecir', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ texto }),
      }))
    } catch (e) { setError(e.message) }
    finally { setOcupado(false) }
  }

  return (
    <>
      <form className={TARJETA} onSubmit={analizar}>
        <label htmlFor="resena" className="mb-2 block font-medium text-titulo">Reseña</label>
        <textarea id="resena" rows={5} maxLength={5000} value={texto}
          placeholder="Por ejemplo: La habitación era amplia y el personal muy atento."
          onChange={(e) => setTexto(e.target.value)}
          className="w-full resize-y rounded-md border border-slate-300 p-3 placeholder:text-slate-400 focus:border-acento focus:ring-2 focus:ring-acento/25 focus:outline-none" />
        <div className="mt-1 text-right text-xs text-slate-400">{texto.length}/5000</div>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span className={NOTA}>Probar con:</span>
          {EJEMPLOS.map((ej) => (
            <button type="button" key={ej} onClick={() => setTexto(ej)}
              className="rounded-full border border-slate-200 bg-fondo-alt px-3 py-1 text-left text-sm text-slate-700 transition-colors hover:border-acento hover:text-acento">{ej}</button>
          ))}
        </div>
        <div className="mt-5 flex gap-2">
          <button disabled={ocupado}
            className="rounded-full bg-acento px-7 py-2.5 font-menu font-medium text-white transition-colors hover:bg-acento-oscuro focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-acento disabled:cursor-wait disabled:opacity-60">
            {ocupado ? 'Analizando…' : 'Analizar reseña'}
          </button>
          {texto && (
            <button type="button" onClick={() => { setTexto(''); setResultado(null); setError('') }}
              className="rounded-full border border-slate-300 px-6 py-2.5 font-menu text-slate-600 hover:border-acento hover:text-acento">Limpiar</button>
          )}
        </div>
        {error && <p role="alert" className="mt-3 text-sm text-negativo">{error}</p>}
      </form>

      {resultado && (
        <div className={`${TARJETA} mt-6`} aria-live="polite">
          {resultado.sentimiento ? (
            <>
              <h3 className="mb-5 flex items-center gap-3 font-titulo text-xl font-bold text-titulo">
                Resultado
                <span className={`rounded-full px-4 py-0.5 font-menu text-base font-medium text-white capitalize ${FONDO_CLASE[resultado.sentimiento]}`}>{resultado.sentimiento}</span>
              </h3>
              <Barras probabilidades={resultado.probabilidades} />
              <p className={`${NOTA} mt-5`}>Las probabilidades son estimaciones del modelo, no una certeza.</p>
            </>
          ) : (
            <>
              <h3 className="mb-3 font-titulo text-xl font-bold text-titulo">Sin clasificar</h3>
              <p className="rounded-md border border-amber-200 bg-amber-50 p-3 text-amber-900">{resultado.aviso}</p>
            </>
          )}
        </div>
      )}
      {resultado && <Preprocesamiento datos={resultado.preprocesamiento} />}
    </>
  )
}

function Metricas({ metricas, error }) {
  if (error) return <div className={TARJETA}><p className="text-negativo">{error}</p></div>
  if (!metricas) return <div className={TARJETA}><p className={NOTA}>Cargando métricas…</p></div>
  const { prueba, validacion_cruzada: cv, modelo_en_produccion: nombre } = metricas
  const cifras = [
    ['Exactitud', porcentaje(prueba.accuracy)],
    ['F1 macro', prueba.f1_macro.toFixed(3)],
    ...CLASES.map((c) => [`F1 ${c}`, prueba.por_clase[c]['f1-score'].toFixed(2)]),
  ]
  return (
    <>
      <p className="mb-4 font-menu text-sm text-slate-600">
        Modelo en producción: <strong className="text-titulo">{nombre}</strong> · evaluación única en el conjunto de prueba (359 reseñas)
      </p>
      {/* Contadores grandes, como la sección "Facts" de la plantilla */}
      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        {cifras.map(([etiqueta, valor]) => (
          <div key={etiqueta} className={`${TARJETA} flex flex-col items-center p-5 text-center`}>
            <dd className="font-titulo text-3xl font-bold text-titulo tabular-nums">{valor}</dd>
            <dt className="mt-1 font-menu text-sm text-slate-500">{etiqueta}</dt>
          </div>
        ))}
      </dl>
      <div className={`${TARJETA} mt-6`}>
        <h3 className="font-titulo text-xl font-bold text-titulo">Comparación de modelos</h3>
        <p className={`${NOTA} mt-1`}>Validación cruzada estratificada de 5 particiones, solo con el conjunto de entrenamiento.</p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-acento text-left font-menu text-titulo">
                <th className="py-2 pr-4 font-medium">Modelo</th>
                <th className="px-2 py-2 text-right font-medium">Exactitud</th>
                <th className="px-2 py-2 text-right font-medium">F1 macro</th>
                <th className="py-2 pl-2 text-right font-medium">F1 neutro</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {cv.map((fila) => {
                const elegido = fila.Modelo === nombre
                return (
                  <tr key={fila.Modelo} className={elegido ? 'bg-acento/5 font-semibold text-titulo' : ''}>
                    <td className="py-3 pr-4 whitespace-nowrap">{fila.Modelo}{elegido && <span className="ml-2 rounded-full bg-acento px-2 py-0.5 text-xs font-medium text-white">en producción</span>}</td>
                    <td className="px-2 py-3 text-right tabular-nums">{fila.Accuracy.toFixed(3)}</td>
                    <td className="px-2 py-3 text-right tabular-nums">{fila['F1 macro'].toFixed(3)}</td>
                    <td className="py-3 pl-2 text-right tabular-nums">{fila['F1 neutro'].toFixed(3)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}

function Funcionamiento() {
  const pasos = [
    ['Normalización', 'Minúsculas, sin tildes (se conserva la ñ), sin URLs, números ni signos; alargamientos reducidos («buenooo» → «bueno»).', 'mensaje'],
    ['Palabras vacías', 'Se quitan artículos, preposiciones y pronombres, pero se conservan negaciones e intensificadores («no», «nunca», «muy», «poco»).', 'cerrar'],
    ['Stemming', 'Cada palabra se reduce a su raíz con el algoritmo Snowball («recomiendo» → «recom»), para que distintas formas cuenten como la misma.', 'codigo'],
    ['TF-IDF', 'Cada reseña se convierte en un vector con el peso de cada raíz en la reseña frente al resto del corpus.', 'grafico'],
    ['Clasificación', 'Una regresión logística con pesos por clase estima la probabilidad de negativo, neutro y positivo.', 'info'],
    ['Datos', 'Corpus COAH (SINAI, Universidad de Jaén): 1.792 reseñas sin duplicados; 1.433 para entrenar y 359 para la prueba final. Etiqueta por estrellas: 1–2 negativo, 3 neutro, 4–5 positivo.', 'libro'],
  ]
  return (
    <>
      {/* Tarjetas con icono, como la sección "Services" de la plantilla */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {pasos.map(([titulo, detalle, icono]) => (
          <div key={titulo} className="group flex gap-4">
            <span className="grid size-14 shrink-0 place-items-center rounded-full bg-acento text-white ring-4 ring-acento/20 transition-colors group-hover:bg-white group-hover:text-acento group-hover:ring-acento">
              <Icono nombre={icono} className="size-6" />
            </span>
            <div>
              <h3 className="font-titulo text-lg font-bold text-titulo">{titulo}</h3>
              <p className="mt-1 text-sm leading-relaxed text-slate-600">{detalle}</p>
            </div>
          </div>
        ))}
      </div>
      <p className={`${NOTA} mt-8`}>Limitación: la clase neutra es la más difícil; en la revisión manual del equipo coincidió solo en un 80 % de los casos.</p>
    </>
  )
}

function Portada({ apiActiva }) {
  const palabra = useEscritura(PALABRAS_PORTADA)
  const estado = apiActiva === null ? ['bg-slate-400', 'Conectando con la API…']
    : apiActiva ? ['bg-positivo', 'API activa'] : ['bg-negativo', 'API sin conexión']
  return (
    // Si existe frontend/public/portada-suite.jpg se usa como fondo; si no, queda el degradado
    <section id="inicio" className="relative flex min-h-screen items-center bg-lateral bg-cover bg-center px-6 sm:px-10 lg:px-24"
      style={{ backgroundImage: "linear-gradient(rgba(4,11,20,.62), rgba(4,11,20,.62)), url('/portada-suite.jpg'), radial-gradient(circle at 75% 30%, rgba(20,157,221,.45), transparent 55%), radial-gradient(circle at 20% 85%, rgba(47,148,97,.25), transparent 50%), linear-gradient(135deg, #0b1d33, #040b14)" }}>
      <div className="max-w-3xl text-white">
        <h1 className="font-titulo text-4xl font-bold leading-tight sm:text-6xl">Sentimiento de reseñas de hoteles</h1>
        <p className="mt-5 text-xl sm:text-2xl">
          Detecto reseñas <span className="border-b-2 border-acento pb-0.5">{palabra}</span>
          <span className="cursor-escritura ml-0.5 border-r-2 border-white" aria-hidden="true">&nbsp;</span>
        </p>
        <p className="mt-6 max-w-xl text-white/75">Procesamiento de lenguaje natural y machine learning sobre 1.792 reseñas reales en español.</p>
        <div className="mt-10 flex flex-wrap items-center gap-6">
          <a href="#analizar" className="inline-flex items-center gap-2 rounded-full bg-acento px-7 py-3 font-menu font-medium text-white transition-colors hover:bg-acento-oscuro">
            Analizar una reseña <Icono nombre="flecha" className="size-4" />
          </a>
          <span className="flex items-center gap-2 font-menu text-sm text-white/80">
            <span className={`size-2.5 rounded-full ${estado[0]}`} />{estado[1]}
          </span>
        </div>
      </div>
    </section>
  )
}

function App() {
  const [menuAbierto, setMenuAbierto] = useState(false)
  const [apiActiva, setApiActiva] = useState(null)
  const [metricas, setMetricas] = useState(null)
  const [errorMetricas, setErrorMetricas] = useState('')
  const activa = useSeccionVisible(SECCIONES.map((s) => s.id))

  useEffect(() => {
    pedir('/salud').then(() => setApiActiva(true)).catch(() => setApiActiva(false))
    pedir('/metricas').then(setMetricas).catch((e) => setErrorMetricas(e.message))
  }, [])

  useEffect(() => {
    const cerrarConEscape = (e) => { if (e.key === 'Escape') setMenuAbierto(false) }
    window.addEventListener('keydown', cerrarConEscape)
    return () => window.removeEventListener('keydown', cerrarConEscape)
  }, [])

  const cerrarEnMovil = () => { if (!esEscritorio()) setMenuAbierto(false) }

  return (
    <div className="min-h-screen bg-white font-sans text-slate-700 antialiased">
      {/* Botón flotante de hamburguesa (solo en pantallas pequeñas, como en la plantilla) */}
      <button aria-label={menuAbierto ? 'Cerrar menú' : 'Abrir menú'} aria-expanded={menuAbierto} aria-controls="menu-lateral"
        onClick={() => setMenuAbierto(!menuAbierto)}
        className="fixed top-4 right-4 z-50 grid size-11 place-items-center rounded-full bg-acento text-white shadow-lg transition-colors hover:bg-acento-oscuro xl:hidden">
        <Icono nombre={menuAbierto ? 'cerrar' : 'menu'} className="size-6" />
      </button>

      <header id="menu-lateral"
        className={`fixed inset-y-0 left-0 z-40 flex w-[300px] flex-col overflow-y-auto bg-lateral px-4 py-6 transition-transform duration-300 xl:translate-x-0 ${menuAbierto ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex flex-col items-center">
          <img src="/icono-5estrellas.jpg" alt="Cinco estrellas doradas sobre un arco" width="120" height="120"
            className="size-[120px] rounded-full border-8 border-lateral-borde object-cover" />
          <p className="mt-4 text-center font-titulo text-2xl font-bold text-white">Sentimiento</p>
          <p className="font-menu text-sm text-lateral-texto">Reseñas de hoteles</p>
          <div className="mt-5 flex gap-2">
            {ENLACES.map((e) => (
              <a key={e.href} href={e.href} target="_blank" rel="noreferrer" title={e.titulo} aria-label={e.titulo}
                className="grid size-9 place-items-center rounded-full bg-lateral-suave text-white transition-colors hover:bg-acento">
                <Icono nombre={e.icono} className="size-[18px]" />
              </a>
            ))}
          </div>
        </div>
        <nav className="mt-8">
          <ul className="space-y-1">
            {SECCIONES.map((s) => {
              const actual = s.id === activa
              return (
                <li key={s.id}>
                  <a href={`#${s.id}`} onClick={cerrarEnMovil} aria-current={actual ? 'location' : undefined}
                    className={`group flex items-center gap-3 px-3 py-3 font-menu text-[15px] transition-colors ${actual ? 'text-white' : 'text-lateral-texto hover:text-white'}`}>
                    <Icono nombre={s.icono} className={`size-6 transition-colors ${actual ? 'text-acento' : 'text-lateral-texto group-hover:text-acento'}`} />
                    {s.titulo}
                  </a>
                </li>
              )
            })}
          </ul>
        </nav>
        <p className="mt-auto pt-8 text-center font-menu text-xs text-lateral-texto/70">Tópicos Avanzados de Programación · UAGRM</p>
      </header>
      <div onClick={() => setMenuAbierto(false)} aria-hidden="true"
        className={`fixed inset-0 z-30 bg-lateral/60 backdrop-blur-sm xl:hidden ${menuAbierto ? 'block' : 'hidden'}`} />

      <main className="xl:ml-[300px]">
        <Portada apiActiva={apiActiva} />
        <Seccion id="analizar">
          <TituloSeccion descripcion="Escribe una reseña de hotel en español. El modelo estima si es negativa, neutra o positiva y muestra cómo procesó el texto.">Analizar reseña</TituloSeccion>
          <Analizar />
        </Seccion>
        <Seccion id="metricas" alterna>
          <TituloSeccion descripcion="Rendimiento del modelo en producción y comparación con el otro modelo candidato y la línea base.">Métricas del modelo</TituloSeccion>
          <Metricas metricas={metricas} error={errorMetricas} />
        </Seccion>
        <Seccion id="funcionamiento">
          <TituloSeccion descripcion="El mismo pipeline de NLP se usa para entrenar el modelo y para responder cada consulta.">Cómo funciona</TituloSeccion>
          <Funcionamiento />
        </Seccion>
        <footer className="bg-fondo-alt px-6 py-6 text-center font-menu text-sm text-slate-500">
          Proyecto integrador · Tópicos Avanzados de Programación · UAGRM
        </footer>
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)
