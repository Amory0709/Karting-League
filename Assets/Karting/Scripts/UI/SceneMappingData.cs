using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace KartGame.UI
{
    [CreateAssetMenu(fileName = "SceneMappingData", menuName = "Custom/Scene Mapping Data")]
    public class SceneMappingData : ScriptableObject
    {
        [SerializeField] private List<SceneMapping> mappings = new List<SceneMapping>();

        public Dictionary<string, string> GetSceneMappings()
        {
            Dictionary<string, string> sceneMappings = new Dictionary<string, string>();
            foreach (SceneMapping mapping in mappings)
            {
                if (!sceneMappings.ContainsKey(mapping.displayName))
                {
                    sceneMappings.Add(mapping.displayName, mapping.sceneName);
                }
                else
                {
                    Debug.LogWarning("Duplicate display name found in scene mappings: " + mapping.displayName);
                }
            }
            return sceneMappings;
        }
    }

    [System.Serializable]
    public class SceneMapping
    {
        public string displayName;
        public string sceneName;
    }
}