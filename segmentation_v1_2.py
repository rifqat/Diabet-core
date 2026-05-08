# Tirsak" (Elbow) metodi yordamida delta uchun optimal qiymatni topadi.
#  23.09.2025

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
import heapq
from kneed import KneeLocator


class YadroSegmentation:
	"""
	Tasvir klasterlash (segmentatsiya) jarayonini yakka klass ichida
	OOP tamoyillariga asoslangan holda amalga oshiruvchi klass.
	"""

	def __init__(self, data, epsilon=0.8, metric="auto"):
		"""
		:param data: Masalan, 2D masofalar uchun (N x 2) o'lchamdagi numpy array
		:param epsilon: Chek vazni (threshold) grafiga qirra qo'shishda ishlatiladi
		:param metric: masofani o'lchash turi. "auto" (o'lcham>10 -> cosine kabi ishlaydi), "euclidean", "cosine" yoki KNN da ishlatiladigan ixtiyoriy.
		"""
		self.data = data
		self.epsilon = epsilon
		self.metric = metric
		
		# Ko'p o'lchovli ma'lumotlar uchun metric ni avto-tanlash
		if self.metric == "auto":
		    self.metric = 'cosine' if self.data.shape[1] > 10 else 'euclidean'

		self.G = None            # Yaratiladigan graf
		self.Dt_sequence = None  # Zichlik o'zgarishlar ketma-ketligi
		self.Mt_sequence = None  # Eng zaif tugunlar ketma-ketligi
		self.labels_ = None  # To store the cluster labels after fitting
		self.lcp = None
		self.core_segments = None  # To store core segments
		self.segments_ = None
		
        # Store the actually used parameters for traceability
		self.used_delta = None
		self.used_beta = None

	def auto_select_epsilon(self, k: int = 5, min_cluster_ratio: float = 0.08) -> float:
		"""
		Hybrid yondashuv bilan epsilon ni avtomatik tanlaydi.

		Ikki strategiya kombinatsiyasi:
		  1. Single-linkage: oxirgi ikkita katta klaster birlashadigan nuqta
		     → yaqin/o'ralgan klasterlar uchun (masalan, doiralar, oylar)
		  2. K-vazn 5-persentil: zich ajralgan klasterlar uchun (masalan, blobs)

		Tanlash qoidasi: birlashuv masofasi (dissimilarity) 0.20 dan kichik bo'lsa
		single-linkage, aks holda k-vazn persentil ishlatiladi.

		Parameters
		----------
		k : int
		    K-vazn hisobida qo'llaniladigan eng yaqin qo'shnilar soni (default 5).
		min_cluster_ratio : float
		    "Katta klaster" deb hisoblash uchun minimal o'lcham ulushi (default 0.08).

		Returns
		-------
		float
		    Tanlangan epsilon qiymati [0.50, 0.98].
		"""
		from scipy.spatial.distance import pdist, squareform
		from scipy.cluster.hierarchy import linkage as scipy_linkage

		N = self.data.shape[0]
		A = squareform(pdist(self.data, metric=self.metric))
		max_vals = A.max(axis=1, keepdims=True)
		W = np.where(max_vals > 0, (max_vals - A) / max_vals, 0.0)
		np.fill_diagonal(W, 0.0)

		min_sz = max(5, int(N * min_cluster_ratio))

		# ── 1. Single-linkage: oxirgi ikkita katta klaster birlashish nuqtasi ──
		cond = (1.0 - W)[np.triu_indices(N, k=1)]
		Z = scipy_linkage(cond, method='single')

		sizes = np.ones(2 * N - 1, dtype=int)
		for i, (c1, c2, d, _) in enumerate(Z):
			sizes[N + i] = sizes[int(c1)] + sizes[int(c2)]

		link_eps   = 0.92
		merge_diss = 1.0     # klasterlar orasidagi masofaning proxisyasi
		for i in range(len(Z) - 1, -1, -1):
			c1, c2, d, _ = Z[i]
			if sizes[int(c1)] >= min_sz and sizes[int(c2)] >= min_sz:
				link_eps   = float(np.clip(1.0 - d + 0.01, 0.50, 0.98))
				merge_diss = float(d)
				break

		# ── 2. K-vazn 5-persentil (zich ajralgan klasterlar uchun) ──────────
		k_weights = [
			np.sort(W[i])[::-1][k - 1]
			for i in range(N)
			if len(W[i]) >= k
		]
		kw_pct = float(np.percentile(k_weights, 5)) if k_weights else 0.92

		# ── Hybrid qaror ─────────────────────────────────────────────────────
		#
		# merge_diss < 0.20 → klasterlar yaqin/o'ralgan (doiralar, oylar)
		#   link_eps < 0.93 → aniq inter-klaster chegara → link_eps ishlatiladi
		#   link_eps ≥ 0.93 → ehtimol shovqin "ko'priklari" → 0.92 (xavfsiz default)
		#
		# merge_diss ≥ 0.20 → klasterlar aniq ajralgan (blobs)
		#                    → k-vazn 5-persentil (0.78 dan kam bo'lmasin)
		if merge_diss < 0.20:
			eps = link_eps if link_eps < 0.93 else 0.92
		else:
			eps = max(kw_pct, 0.78)

		self.epsilon = float(np.clip(eps, 0.50, 0.98))
		return self.epsilon

	def create_graph_with_weights(self):
		"""
		O(N log N) tezlikda NearestNeighbors orqali graf yaratish va qirra vaznini (threshold) belgilash.
		Katta ma'lumotlarda tezlikni million marta oshiradi.
		"""
		from sklearn.neighbors import NearestNeighbors
		self.G = nx.Graph()
		self.G.add_nodes_from(range(len(self.data)))
		
		# KNN algoritmi orqali faqat potentsial yaqin nuqtalarni qidirish
		metric = self.metric
		nn = NearestNeighbors(n_neighbors=min(500, len(self.data)-1), metric=metric, n_jobs=-1)
		nn.fit(self.data)
		distances, indices = nn.kneighbors(self.data)
		
		# Maksimal uzoqlikni topish (og'irliklarni normalize qilish uchun)
		# Bu yerda biz maksimal masofani global yoki har bir nuqta uchun hisoblashimiz mumkin
		# YadroSeg mantig'iga ko'ra lokal (har bir nuqtaning eng uzoq masofasi) ko'proq foyda beradi,
		# lekin qidiruv radiusini hisobga olgan holda max masofani k-chi qo'shni masofasi sifatida olamiz
		max_dists = distances[:, -1].reshape(-1, 1) + 1e-10  # 0 ga bo'lish xatoligi oldini olish
		
		# Vaznlarni (max - dist) / max formula bilan hisoblash
		weights = np.where(max_dists > 0, (max_dists - distances) / max_dists, 0)
		
		edges_to_add = []
		for u in range(len(self.data)):
			for i, v in enumerate(indices[u]):
				if u < v and weights[u, i] > self.epsilon:
					edges_to_add.append((u, v, weights[u, i]))
					
		self.G.add_weighted_edges_from(edges_to_add)

	def find_max_min_neighbors(self):
		"""
		Grafiga asoslangan holda maksimal va minimal darajali tugunlarni topish.
		Izolyatsiyalangan tugunlar ro‘yxatini ham qaytaradi.
		"""
		if self.G is None:
			raise ValueError("Graf hali yaratilmagan. create_graph_with_weights() chaqiring.")

		if len(self.G.nodes) == 0:
			return None, 0, None, 0, []

		degrees = dict(self.G.degree())
		max_degree_node = max(degrees, key=degrees.get)
		min_degree_node = min(degrees, key=degrees.get)
		max_degree = degrees[max_degree_node]
		min_degree = degrees[min_degree_node]

		isolated_nodes = [node for node, deg in self.G.degree() if deg == 0]

		return max_degree_node, max_degree, min_degree_node, min_degree, isolated_nodes

	def compute_local_density(self, G):
		"""
		Tugunlarning qo‘shnilari bilan qirra vaznlarini yig‘indi sifatida
		lokal zichlik (density) qiymatini qaytaradi.
		"""
		# networkx dagi C-tilida optimallashtirilgan ichki usul orqali tezkor hisoblash
		return dict(G.degree(weight='weight'))

	def compute_density_variation_sequence(self):
		"""
		Zichlik o'zgarishlar ketma-ketligini (Dt_sequence) va eng zaif tugunlar ketma-ketligini (Mt_sequence) heapq
		yordamida samarali hisoblaydi. Bu implementatsiya har bir qadamda heapni qayta qurishdan saqlanadi,
		bu esa katta graflar uchun ishlash tezligini sezilarli darajada oshiradi.
		"""
		if self.G is None:
			raise ValueError("Graf hali yaratilmagan. create_graph_with_weights() chaqiring.")

		# Lokal zichliklarni boshlang‘ich hisoblash
		density = self.compute_local_density(self.G)
		# Heap (min-heap) zichlik va tugun juftliklaridan iborat
		heap = [(density[node], node) for node in self.G.nodes()]
		heapq.heapify(heap)

		# Qaysi tugunlar allaqachon ketma-ketlikka qo'shilganini kuzatish
		processed_nodes = set()

		Dt_sequence = []
		Mt_sequence = []

		while heap:
			# Heapdan eng kichik zichlikka ega tugunni olamiz
			Dt, Mt = heapq.heappop(heap)

			# Agar bu tugun allaqachon qayta ishlangan bo'lsa (heapdagi eski yozuv), o'tkazib yuboramiz
			if Mt in processed_nodes:
				continue

			# Tugunni qayta ishlangan deb belgilaymiz va ketma-ketliklarga qo'shamiz
			processed_nodes.add(Mt)
			Dt_sequence.append(Dt)
			Mt_sequence.append(Mt)

			# Qo‘shnilar zichligini yangilash
			for nbr in self.G.neighbors(Mt):
				# Faqat hali qayta ishlanmagan qo'shnilarni yangilaymiz
				if nbr not in processed_nodes:
					# Manfiy tomonga o'tib ketmasligini oldini olish (precision float errors)
					density[nbr] = max(0.0, density[nbr] - self.G[Mt][nbr]['weight'])
					# Yangilangan zichlik bilan qo'shnini heapga qayta qo'shamiz
					heapq.heappush(heap, (density[nbr], nbr))

		self.Dt_sequence = Dt_sequence
		self.Mt_sequence = Mt_sequence

		return Dt_sequence, Mt_sequence

	def identify_core_pixels(self, Dt_sequence, Mt_sequence, delta, beta):
		"""
		Zichlik pasayish tezligi R_t ga asoslangan holda asosiy (core) tugunlarni aniqlash.
		:param Dt_sequence: Zichliklar ketma-ketligi
		:param Mt_sequence: Eng zaif tugunlar ketma-ketligi
		:param delta: R_t ni saralashdagi nisbiy threshold (masalan, 0.5)
		:param beta: Ketma-ket R_t > alpha bo'lgan minimal son (masalan, 5)
		:return: core_pixels ro‘yxati
		"""
		# R_t = (D_t - D_(t+1)) / D_t
		# NumPy orqali vektorlashtirilgan hisob-kitob (katta massivlar uchun tezkor)
		Dt_array = np.array(Dt_sequence)
		with np.errstate(divide='ignore', invalid='ignore'):
			Rt_array = np.where(Dt_array[:-1] != 0, (Dt_array[:-1] - Dt_array[1:]) / Dt_array[:-1], 0)
		Rt_sequence = Rt_array.tolist()
		# Dt_sequence dan biri kam chiqqani uchun oxiriga 0 qo'shamiz
		Rt_sequence.append(0)

		# Ijobiy R_t larni saralab, sorted qilamiz
		positive_Rt = [r for r in Rt_sequence if r > 0]
		if not positive_Rt:
			return []

		positive_Rt_sorted = sorted(positive_Rt)
		alpha_index = int(len(positive_Rt_sorted) * delta)
		if alpha_index >= len(positive_Rt_sorted):
			alpha_index = len(positive_Rt_sorted) - 1

		alpha = positive_Rt_sorted[alpha_index]
		print(f"Alpha (delta={delta} ga asoslangan): {alpha:.4f}")

		# Asosiy piksellar to'plamini aniqlash
		core_pixels = []
		consecutive_count = 0

		for t in range(len(Rt_sequence)):
			if Rt_sequence[t] > alpha:
				consecutive_count += 1
				if consecutive_count >= beta:
					core_pixels.append(Mt_sequence[t])
			else:
				consecutive_count = 0

		return core_pixels

	def partition_core_pixels(self, G, core_pixels, theta=0.1):
		"""
		Asosiy tugunlardan subgraf (core_graph) tuzib, past vaznli qirralar (w < theta) ni olib tashlaydi
		va ulanmagan komponentlarni topadi.
		"""
		core_graph = G.subgraph(core_pixels).copy()
		# Zaif qirralarni olib tashlash
		weak_edges = [(u, v) for u, v, w in core_graph.edges(data='weight') if w < theta]
		core_graph.remove_edges_from(weak_edges)

		# Ulanmagan komponentlarni (connected components) topish
		segments = list(nx.connected_components(core_graph))
		return segments

	def calculate_similarity(self, G, pixel, segment):
		"""
		Berilgan tugun (pixel) va segment orasidagi o‘rtacha vazn (similarity) ni hisoblash.
		"""
		weights = [G[pixel][s]['weight'] for s in segment if G.has_edge(pixel, s)]
		if weights:
			return np.mean(weights)
		return 0

	def expand_segments(self, G, Mt_sequence, core_segments, lambda_value=0.5):
		"""
		Orqaga ketma-ketlikda (Mt_sequence) tugunlarni klasterlarga qo‘shish (segmentlarni kengaytirish).
		:return: (segments, low_confidence_pixels)
		"""
		segments = [list(s) for s in core_segments if s] # Faqat bo'sh bo'lmagan segmentlarni olamiz
		low_confidence_pixels = set()
		# Asosiy (core) piksellarni boshidanoq qo'shilgan deb hisoblaymiz
		added_pixels = {node for seg in segments for node in seg}

		# Har bir qo'shilgan piksel qaysi segmentga tegishli ekanligini saqlaydigan dict (Fast Lookup O(1))
		pixel_to_segment_idx = {}
		for i, seg in enumerate(segments):
			for node in seg:
				pixel_to_segment_idx[node] = i

		n = len(Mt_sequence)
		# Orqadan oldinga qarab (t = n-1 -> 0)
		for t in range(n - 1, -1, -1):
			Mt = Mt_sequence[t]

			if Mt not in added_pixels:
				segment_weights = {}
				
				# Faqatgina shu tugunning haqiqiy qo'shnilarini graf orqali aylanib chiqamiz
				for nbr in G.neighbors(Mt):
					if nbr in pixel_to_segment_idx:
						seg_idx = pixel_to_segment_idx[nbr]
						if seg_idx not in segment_weights:
							segment_weights[seg_idx] = []
						segment_weights[seg_idx].append(G[Mt][nbr]['weight'])

				if segment_weights:
					# Har bir ulangan segment bilan o'rtacha o'xshashlik hisoblanadi
					similarities = []
					for seg_idx, weights in segment_weights.items():
						similarities.append((seg_idx, np.mean(weights)))
					
					similarities.sort(key=lambda x: x[1], reverse=True)
					s1, m1 = similarities[0]
					m2 = similarities[1][1] if len(similarities) > 1 else 0

					# Eng o‘xshash segmentga qo‘shish
					if m1 > 0:
						segments[s1].append(Mt)
						added_pixels.add(Mt)
						pixel_to_segment_idx[Mt] = s1 # Yangi tugun ro'yxatga olindi
						
						# Soft assignment: ko'p o'lchovlilar uchun qattiqqo'llikni pasaytiramiz
						if m2 > lambda_value * m1:
							low_confidence_pixels.add(Mt)
				else:
					# Ba'zi shovqin holatlarida eng yaqin tugunning manzilini yagona cluster deb ulaymiz (Soft-noise reduction)
					# Hech qanday ulangan "core" klasteri topilmasa, u uzilib qoladi va aslida shovqin bo'lishi ehtimoli yuqori.
					new_seg_idx = len(segments)
					segments.append([Mt])
					added_pixels.add(Mt)
					pixel_to_segment_idx[Mt] = new_seg_idx

		# Yuqoridagi mantiq bo'yicha har bir tugun faqat bir marta qo'shiladi,
		# shuning uchun `set()` orqali takrorlanishlarni olib tashlashga hojat yo'q.

		# Shuningdek, `Mt_sequence` grafiga tegishli barcha tugunlarni o'z ichiga oladi va
		# har bir tugun qayta ishlanganligi sababli, `missing_nodes` ro'yxati bo'sh bo'lishi kerak.
		# Quyidagi kod bloki ehtiyot choralari sifatida qoldirilgan, ammo odatda ishlamasligi kerak.
		# all_nodes = set(G.nodes)
		# added_nodes = {node for seg in segments for node in seg}
		# missing_nodes = all_nodes - added_nodes
		# if missing_nodes:
		# 	# Bu holat yuzaga kelmasligi kerak. Agar kelsa, bu logik xato.
		# 	segments.extend([[node] for node in missing_nodes])
		return segments, low_confidence_pixels



	def find_optimal_params_elbow(self, beta=5, visualize_elbow=True):
		"""
		YANGILANGAN: "Tirsak" (Elbow) metodi yordamida delta uchun optimal qiymatni topadi.
		Bu versiya "chiziqdan eng uzoq nuqta" yondashuvidan foydalanadi, bu esa barqarorroq.

		:param beta: Beta uchun fiksatsiyalangan qiymat.
		:param visualize_elbow: Tirsakni topish jarayonini vizualizatsiya qilish uchun bayroq.
		:return: (optimal_delta, beta)
		"""
		print("Optimal delta'ni 'chiziqdan eng uzoq nuqta' metodi yordamida izlash...")
		if self.Dt_sequence is None or self.Mt_sequence is None:
			self.create_graph_with_weights()
			self.compute_density_variation_sequence()

		dt_seq = np.array(self.Dt_sequence)
		with np.errstate(divide='ignore', invalid='ignore'):
			Rt_sequence = (dt_seq[:-1] - dt_seq[1:]) / dt_seq[:-1]
		Rt_sequence[~np.isfinite(Rt_sequence)] = 0
		
		positive_Rt = sorted([r for r in Rt_sequence if r > 0])
		
		if len(positive_Rt) < 3:
			print("Tirsakni aniqlash uchun yetarli R_t qiymatlari topilmadi. Standart delta=0.5 qaytarilmoqda.")
			return 0.5, beta

		# --- Yangi mantiq: Chiziqdan eng uzoq nuqtani topish ---
		# Kneedle yondashuvi uchun X va Y ni [0, 1] oralig'iga normallashtirish muhim
		y_points = np.array(positive_Rt)
		x_points = np.arange(len(y_points))
		
		# Normallashtirish (X va Y masshtabi har xil bo'lsa xato ishlaydi, shuning uchun [0,1] ga tushiramiz)
		x_norm = (x_points - x_points[0]) / (x_points[-1] - x_points[0] + 1e-10)
		y_norm = (y_points - y_points[0]) / (y_points[-1] - y_points[0] + 1e-10)

		# Boshlang'ich (0,0) va oxirgi (1,1) nuqtalarni bog'laydigan chiziq
		p1 = np.array([0.0, 0.0])
		p2 = np.array([1.0, 1.0])

		line_vec = p2 - p1
		line_vec_norm = np.linalg.norm(line_vec)
		
		vec_from_p1 = np.column_stack((x_norm, y_norm)) - p1
		cross_product = np.cross(vec_from_p1, line_vec)
		
		# Qavariq (convex) egrilik bo'lgani uchun, diagonal ostidagi eng uzoq masofani olamiz.
		# Aslida `x_norm - y_norm` ning o'zi masofaga proporsional.
		distances = np.abs(cross_product) / line_vec_norm

		# Eng uzoq nuqtani (tirsakni) topish
		opt_idx = np.argmax(distances)
		optimal_delta = opt_idx / len(positive_Rt)
		
		# Natijani [0.0, 0.95] oralig'ida cheklaymiz. (Ko'p o'lchovlilar uchun 0.0 ham juda zo'r natija beryapti)
		optimal_delta = max(0.0, min(0.95, optimal_delta))
		
		# Tabular (qiyin ko'p o'lchovli) ma'lumotlarda Kneedle baribir ko'pincha kattalashtirib yuborsa (0.6+), 
		# Biz bilamizki cosine metric da zichlik asosan 0.0 - 0.3 oraliqlarida haqiqiy core larni topadi.
		if getattr(self, 'metric', 'euclidean') == 'cosine':
			# Kosinus o'xshashligida egrilik doim farq qiladi. Agar > 0.15 bo'lsa uni proportion kichraytiramiz
			if optimal_delta > 0.15:
				optimal_delta = optimal_delta * 0.1 # Natija asosan 0.01 - 0.09 gacha tushib keladi
				
		print(f"Topilgan 'tirsak' nuqtasi (indeks {opt_idx}) asosida hisoblangan delta: {optimal_delta:.4f}")

		if visualize_elbow:
			plt.figure(figsize=(10, 6))
			plt.plot(x_points, y_points, 'b-', marker='.', label='Saralangan R_t qiymatlari')
			plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', label='Boshlang\'ich-oxirgi nuqta chizig\'i')
			plt.plot(x_points[opt_idx], y_points[opt_idx], 'go', markersize=10, label=f'Topilgan Tirsak (Delta ≈ {optimal_delta:.2f})')
			plt.title("'Tirsak' Metodi Vizualizatsiyasi")
			plt.xlabel("Saralangan R_t Indeksi")
			plt.ylabel("R_t Qiymati")
			plt.legend()
			plt.grid(True)
			plt.show()

		return optimal_delta, beta
	
	
	def compute_CSI(self, G, core_segments):
		"""(TUZATILGAN) CSI (Core Separation Index) ni hisoblaydi.
		(Katta graflar uchun tugun qidiruvlar Hash-map orqali optimallashtirildi)
		"""
		if not core_segments or len(core_segments) < 2:
			return 0.0

		# Har bir tugun qaysi core_segment ga tegishli ekanligini tezkor bilish uchun Dictionary (Hash-map) yaratamiz
		node_to_core_idx = {}
		for idx, core in enumerate(core_segments):
			for node in core:
				node_to_core_idx[node] = idx

		csi_ratios = []
		for i in range(len(core_segments)):
			core_i = set(core_segments[i])
			subgraph_i = G.subgraph(core_i)
			
			# TUZATILGAN: Zichlikni node soniga nisbatan hisoblash
			num_nodes_i = len(core_i)
			max_possible_edges = num_nodes_i * (num_nodes_i - 1) / 2
			if max_possible_edges > 0:
				cohesion_i = subgraph_i.size(weight='weight') / max_possible_edges
			else:
				cohesion_i = 0

			# Har bir boshqa klaster (j) bilan bog'lanish vaznlarini yig'ish dicti
			inter_cluster_weights = {}

			# Tezkor qidiruv: i-klasterdagi har bir tugunning barcha C(qo'shnilarini) tekshiramiz
			for u in core_i:
				for v in G.neighbors(u):
					# Agar qo'shni (v) qaysidir boshqa core_segment(j) da bo'lsa (O(1) qidiruv)
					if v in node_to_core_idx:
						j = node_to_core_idx[v]
						if i != j:
							if j not in inter_cluster_weights:
								inter_cluster_weights[j] = 0
							inter_cluster_weights[j] += G[u][v]['weight']

			max_coupling = 0  # TUZATILGAN: 0 dan boshlash

			# Tashqi klasterlar bilan maxsimal aralashib ketish proporsiyasini topamiz
			for j in range(len(core_segments)):
				if i == j:
					continue
					
				boundary_weight = inter_cluster_weights.get(j, 0)
				num_possible = num_nodes_i * len(core_segments[j])
				coupling_ij = boundary_weight / num_possible if num_possible > 0 else 0
				max_coupling = max(max_coupling, coupling_ij)

			# TUZATILGAN: max_coupling = 0 bo'lganda to'g'ri ishlov berish
			if max_coupling > 0:
				csi_ratios.append(cohesion_i / max_coupling)
			elif cohesion_i > 0:
				csi_ratios.append(float('inf'))  # Ajratilgan, bog'liq emas
			else:
				csi_ratios.append(0.0)

		# Inf qiymatlarni olib tashlash
		finite_ratios = [r for r in csi_ratios if np.isfinite(r)]
		return np.mean(finite_ratios) if finite_ratios else 0.0