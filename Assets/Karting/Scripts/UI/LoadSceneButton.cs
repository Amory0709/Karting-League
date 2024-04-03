using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace KartGame.UI
{
    public class LoadSceneButton : MonoBehaviour
    {
        public TMP_Dropdown dropdown;
        public SceneMappingData mappingData;

        private Dictionary<string, string> sceneMappings;

        private void Start()
        {
            // Initialize the scene mappings from the ScriptableObject
            if (mappingData != null)
                sceneMappings = mappingData.GetSceneMappings();
        }

        public void LoadTargetScene()
        {
            var sceneName = "IntroMenu";
            if (dropdown != null)
            {
                int selectedOptionIndex = dropdown.value;
                string selectedOption = dropdown.options[selectedOptionIndex].text;

                // Retrieve the corresponding scene name based on the selected option
                sceneName = GetSceneName(selectedOption);
                CrossSceneInfo.Round = sceneName;
            }            
            // Load the scene
            SceneManager.LoadSceneAsync(sceneName);
        }

        private string GetSceneName(string displayName)
        {
            if (sceneMappings.TryGetValue(displayName, out var sceneName))
            {
                return sceneName;
            }

            Debug.LogError("No scene mapping found for display name: " + displayName);
            return "IntroMenu";
        }
    }
}
