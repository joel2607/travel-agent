
def _cluster_places_by_distance(places, distance_matrix, max_daily_distance=10000):
    """Simple greedy clustering: Group places within distance limit."""
    # Assume distance_matrix['rows'][i]['elements'][j]['distance']['value'] gives meters
    clusters = []
    remaining = places.copy()
    
    while remaining:
        cluster = [remaining[0]]
        remaining = remaining[1:]
        total_distance = 0
        
        for place in remaining:
            # Find distance from last in cluster to this place
            last_idx = next(i for i, p in enumerate(places) if p.place_id == cluster[-1].place_id)
            this_idx = next(i for i, p in enumerate(places) if p.place_id == place.place_id)
            dist = distance_matrix['rows'][last_idx]['elements'][this_idx]['distance']['value']
            
            if total_distance + dist <= max_daily_distance:
                cluster.append(place)
                total_distance += dist
            else:
                break
        
        clusters.append(cluster)
    
    return clusters

def _parse_duration_to_days(duration: str) -> int:
    """Convert '1 week' to days."""
    if "week" in duration.lower():
        return 7
    elif "day" in duration.lower():
        return 1
    return 3  # Default

def _basic_travel_plan(state, places, preferences):
    """Fallback for basic plan without optimization."""
    # Original logic here (group by category, generate text)
    pass  # Implement as needed from original

def _generate_basic_narrative(itineraries, preferences, memory_context):
    """Fallback narrative."""
    text = f"**Duration:** {preferences.duration}\n**Budget:** {preferences.budget}\n"
    for itinerary in itineraries:
        text += f"\n**Day {itinerary['day']}:**\n"
        for place in itinerary['places']:
            text += f"- {place.name} ({place.rating}⭐)\n"
        text += f"{itinerary['route']}\n"
    text += f"\n**Memory Notes:** {memory_context}"
    return text
