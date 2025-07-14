from .find_LCS import find_longest_common_sequence

def get_attr(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

def merge_transcripts(results: list[tuple[dict, int]]) -> dict:
    #print("\nMerging results...")

    has_segments = any(
        ('segments' in (chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk))
        for chunk, _ in results
    )

    has_words = False
    words = []

    for chunk, chunk_start_ms in results:
        data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
        chunk_words = data.get('words', []) if isinstance(data, dict) else getattr(chunk, 'words', [])

        if chunk_words:
            has_words = True
            for word in chunk_words:
                word_start = get_attr(word, 'start', 0) + chunk_start_ms / 1000
                word_end = get_attr(word, 'end', 0) + chunk_start_ms / 1000
                words.append({
                    'word': get_attr(word, 'word', ''),
                    'start': word_start,
                    'end': word_end
                })

    if not has_segments:
        texts = [
            get_attr(chunk, 'text', '') if not isinstance(chunk, dict)
            else chunk.get('text', '')
            for chunk, _ in results
        ]
        result = {"text": " ".join(texts), "segments": []}
        if has_words:
            result["words"] = words
        return result

    #print("Merging segments across chunks...")
    final_segments = []
    processed_chunks = []

    for i, (chunk, chunk_start_ms) in enumerate(results):
        data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
        segments = get_attr(data, 'segments', [])
        dict_segments = [{
            'text': get_attr(seg, 'text', ''),
            'start': get_attr(seg, 'start', 0),
            'end': get_attr(seg, 'end', 0)
        } for seg in segments]

        if i < len(results) - 1:
            next_start = results[i + 1][1]
            current_segments, overlap_segments = [], []

            for seg in dict_segments:
                if seg['end'] * 1000 > next_start:
                    overlap_segments.append(seg)
                else:
                    current_segments.append(seg)

            if overlap_segments:
                merged_overlap = {
                    'text': ' '.join(s['text'] for s in overlap_segments),
                    'start': overlap_segments[0]['start'],
                    'end': overlap_segments[-1]['end']
                }
                current_segments.append(merged_overlap)

            processed_chunks.append(current_segments)
        else:
            processed_chunks.append(dict_segments)

    for i in range(len(processed_chunks) - 1):
        if not processed_chunks[i] or not processed_chunks[i + 1]:
            continue

        if len(processed_chunks[i]) > 1:
            final_segments.extend(processed_chunks[i][:-1])

        last_seg = processed_chunks[i][-1]
        first_seg = processed_chunks[i + 1][0]

        merged_text = find_longest_common_sequence([last_seg['text'], first_seg['text']])
        merged_seg = {
            'text': merged_text,
            'start': last_seg['start'],
            'end': first_seg['end']
        }
        final_segments.append(merged_seg)

    if processed_chunks and processed_chunks[-1]:
        final_segments.extend(processed_chunks[-1])

    final_text = ' '.join(seg['text'] for seg in final_segments)
    result = {"text": final_text, "segments": final_segments}
    if has_words:
        result["words"] = words

    return result
